# 03_build_wastewater.py
# =============================================================================
# Purpose: turn the raw wastewater samples into ONE weekly value per state,
# exactly as it could have been known on each Friday forecast date
# (Analysis Plan, section 3, steps 1-8). Smoothing and gap-filling (steps 9-10)
# happen later, inside the forecasting script, because the smoothing window is
# chosen during tuning.
#
# Steps
#   1. Reporting delay per series, measured from the version history
#      (data/raw/nwss_archive): days from sample collection to first release.
#   2. Clean sample table: included states and series only, H5 rule applied,
#      log10 values, and the date each sample became available.
#   3. Weekly value per series = mean of log10 sample values in that epiweek.
#   4. Baseline per series = median weekly value over the tuning period.
#      A series with no tuning-period data enters only after it has 8 weeks of
#      data available, and its baseline is the median of those first 8 weeks.
#   5. State value = median over its series of (weekly value - baseline).
#   6. For every Friday forecast date, keep the last 8 weeks of state values
#      using only samples available by that date ("pseudo-real-time").
#   7. The same table built from the TRUE versions (sensitivity analysis S1),
#      for forecast dates from 27 Feb 2026, and the revision summary (S2).
#
# Reads : data/raw/nwss_latest/, data/raw/nwss_archive/,
#         data/summary/series_lod.csv, data/summary/coverage_by_state.csv
# Writes: data/processed/ww_delays.csv
#         data/processed/ww_samples.csv.gz
#         data/processed/ww_state_asof.csv.gz          (main analysis)
#         data/processed/ww_state_asof_realtime.csv.gz (sensitivity S1)
#         data/processed/ww_revisions.csv              (sensitivity S2)
# Run   : python scripts/03_build_wastewater.py
# =============================================================================

import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
SUMMARY_DIR = os.path.join("data", "summary")
OUT_DIR = os.path.join("data", "processed")

PATHOGENS = ["flu", "rsv", "covid"]

# Tuning period weeks (same as 02_check_coverage.py)
TUNING_FIRST_WEEK_END = pd.Timestamp("2023-10-07")
TUNING_LAST_WEEK_END = pd.Timestamp("2024-11-16")

# Forecast dates: every Friday. Tuning origins start once a few weeks of
# tuning-period data exist; the last origin is the latest Friday in the data.
FIRST_ORIGIN = pd.Timestamp("2023-11-03")
ASOF_WEEKS = 8                      # weeks of history kept per forecast date

# Version history starts 25 Feb 2026. To measure delays without bias we use
# samples collected after that start (so their first release is not cut off)
# and at least 60 days before the last version (so slow samples are not missed).
ARCHIVE_START = pd.Timestamp("2026-02-25")
DELAY_SAMPLE_FIRST = pd.Timestamp("2026-03-01")
MIN_SAMPLES_FOR_OWN_DELAY = 8       # fewer -> use the pathogen-wide median
FIRST_REALTIME_ORIGIN = pd.Timestamp("2026-02-27")   # first Friday after 25 Feb

H5_WINDOW_WEEKS = 4                 # H5 detection in this week or previous 4
NEW_SERIES_BASELINE_WEEKS = 8


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def read_folder(name):
    """Read and stack every monthly file of one dataset."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, name, f"{name}_*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"No files in {RAW_DIR}/{name}. Run 01_download_data.py first.")
    tables = [pd.read_csv(f, dtype={"geo_value": str, "sample_index": str,
                                    "county_fips": str}) for f in files]
    return pd.concat([t for t in tables if len(t) > 0], ignore_index=True)


def week_ending_saturday(dates):
    """Saturday that ends each date's Sunday-Saturday (MMWR) week."""
    dates = pd.to_datetime(dates)
    return dates + pd.to_timedelta((5 - dates.dt.day_of_week) % 7, unit="D")


def report_date(report_time):
    """Delphi report times look like '2026-03-06 12:00:00+00:00'. Keep the calendar date."""
    return pd.to_datetime(report_time, utc=True).dt.tz_localize(None).dt.normalize()


def add_series_columns(df):
    """Add pathogen, series (site x lab) and week_end columns."""
    df["pathogen"] = df["signal"].str.split("_").str[0]
    df["series"] = df["geo_value"] + "|" + df["nwss_source"]
    df["reference_time"] = pd.to_datetime(df["reference_time"])
    df["week_end"] = week_ending_saturday(df["reference_time"])
    return df


# -----------------------------------------------------------------------------
# Step 1: reporting delays from the version history
# -----------------------------------------------------------------------------
def measure_delays(archive):
    """Days from collection to first release, median per (pathogen, series)."""
    archive["report_date"] = report_date(archive["report_time"])
    last_version = archive["report_date"].max()
    first_release = (archive.groupby(["pathogen", "series", "sample_index"])
                     .agg(reference_time=("reference_time", "first"),
                          first_report=("report_date", "min"))
                     .reset_index())
    usable = first_release[
        (first_release["reference_time"] >= DELAY_SAMPLE_FIRST)
        & (first_release["reference_time"] <= last_version - pd.Timedelta(days=60))]
    usable = usable.assign(delay_days=(usable["first_report"] - usable["reference_time"]).dt.days)

    per_series = (usable.groupby(["pathogen", "series"])["delay_days"]
                  .agg(n_samples="size", median_delay="median").reset_index())
    per_pathogen = usable.groupby("pathogen")["delay_days"].median().rename("pathogen_median")
    return per_series, per_pathogen, usable


# -----------------------------------------------------------------------------
# Steps 2-5 for one forecast date
# -----------------------------------------------------------------------------
def state_values_asof(samples, origin, baselines_tuning):
    """State weekly values (minus baseline) using only samples available by `origin`.

    samples          : cleaned sample table with columns pathogen, state, series,
                       week_end, log10_value, available_date
    origin           : the Friday forecast date
    baselines_tuning : tuning-period baseline per (pathogen, series)
    Returns rows (origin, pathogen, state, week_end, ww_value, n_series)
    for the last ASOF_WEEKS weeks before the origin."""
    known = samples[samples["available_date"] <= origin]

    # Step 3: weekly value per series
    weekly = (known.groupby(["pathogen", "state", "series", "week_end"])["log10_value"]
              .mean().rename("weekly").reset_index())

    # Step 4a: series with a tuning-period baseline
    weekly = weekly.merge(baselines_tuning, on=["pathogen", "series"], how="left")

    # Step 4b: new series -> baseline from their first 8 weeks, and they only
    # count once those 8 weeks are available
    new = weekly["baseline"].isna()
    if new.any():
        nw = weekly[new].sort_values("week_end")
        nw["rank"] = nw.groupby(["pathogen", "series"]).cumcount() + 1
        first8 = nw[nw["rank"] <= NEW_SERIES_BASELINE_WEEKS]
        base_new = (first8.groupby(["pathogen", "series"])
                    .agg(base_new=("weekly", "median"), n=("weekly", "size")))
        base_new = base_new[base_new["n"] == NEW_SERIES_BASELINE_WEEKS]["base_new"].reset_index()
        weekly = weekly.merge(base_new, on=["pathogen", "series"], how="left")
        weekly["baseline"] = weekly["baseline"].fillna(weekly["base_new"])
        weekly = weekly.drop(columns="base_new")
    weekly = weekly.dropna(subset=["baseline"])
    weekly["relative"] = weekly["weekly"] - weekly["baseline"]

    # Step 5: state value = median over series; keep the last ASOF_WEEKS weeks
    window_start = origin - pd.Timedelta(weeks=ASOF_WEEKS)
    weekly = weekly[(weekly["week_end"] > window_start) & (weekly["week_end"] < origin)]
    state = (weekly.groupby(["pathogen", "state", "week_end"])["relative"]
             .agg(ww_value="median", n_series="size").reset_index())
    state.insert(0, "origin", origin)
    return state


def tuning_baselines(samples):
    """Median weekly log10 value per (pathogen, series) over the tuning period.
    Uses all tuning-period samples: the tuning period ends before the first
    evaluation forecast, so no evaluation-period information enters."""
    in_tuning = samples["week_end"].between(TUNING_FIRST_WEEK_END, TUNING_LAST_WEEK_END)
    weekly = (samples[in_tuning].groupby(["pathogen", "series", "week_end"])["log10_value"]
              .mean().reset_index())
    return (weekly.groupby(["pathogen", "series"])["log10_value"].median()
            .rename("baseline").reset_index())


def fridays(first, last):
    return pd.date_range(first, last, freq="W-FRI")


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- inclusion decisions from 02_check_coverage.py ----------
    lod = pd.read_csv(os.path.join(SUMMARY_DIR, "series_lod.csv"), dtype={"series": str})
    cov = pd.read_csv(os.path.join(SUMMARY_DIR, "coverage_by_state.csv"))
    kept_series = lod.loc[~lod["excluded_lod"], ["pathogen", "series"]]
    kept_states = cov.loc[cov["passes_main_rule"], ["pathogen", "state"]]

    # ---------- latest wastewater ----------
    print("Reading latest wastewater files ...")
    ww = add_series_columns(read_folder("nwss_latest"))
    ww["state"] = ww["state_territory"].str.lower()
    site_state = ww[["geo_value", "state"]].drop_duplicates("geo_value")

    # ---------- Step 1: delays ----------
    print("Reading version history and measuring reporting delays ...")
    arc = add_series_columns(read_folder("nwss_archive"))
    per_series, per_pathogen, delay_samples = measure_delays(arc)
    delays = per_series.merge(per_pathogen.reset_index(), on="pathogen", how="left")
    own = delays["n_samples"] >= MIN_SAMPLES_FOR_OWN_DELAY
    delays["delay_used"] = np.where(own, delays["median_delay"], delays["pathogen_median"])
    delays["delay_source"] = np.where(own, "own", "pathogen_median")
    delays.to_csv(os.path.join(OUT_DIR, "ww_delays.csv"), index=False)

    # ---------- Step 2: clean samples ----------
    # H5 flags: any lab, by site and week (flu only)
    h5 = ww[(ww["signal"] == "flu_h5_avg_conc") & (ww["value"] > 0)]
    h5_weeks = h5[["geo_value", "week_end"]].drop_duplicates()
    h5_flag = pd.concat([h5_weeks.assign(week_end=h5_weeks["week_end"] + pd.Timedelta(weeks=k))
                         for k in range(H5_WINDOW_WEEKS + 1)]).drop_duplicates()
    h5_flag["h5_flag"] = True

    s = ww[ww["signal"].isin([f"{p}_avg_conc_lin" for p in PATHOGENS])].copy()
    s = s.merge(kept_series, on=["pathogen", "series"], how="inner")
    s = s.merge(kept_states, on=["pathogen", "state"], how="inner")
    s = s[s["value"] > 0]                                      # log needs positive values
    s = s.merge(h5_flag, on=["geo_value", "week_end"], how="left")
    s["h5_flag"] = s["h5_flag"].fillna(False).astype(bool) & (s["pathogen"] == "flu")
    n_h5 = int(s["h5_flag"].sum())
    s = s[~s["h5_flag"]]

    s = s.merge(delays[["pathogen", "series", "delay_used"]], on=["pathogen", "series"], how="left")
    s["delay_used"] = s["delay_used"].fillna(s["pathogen"].map(per_pathogen))
    s["available_date"] = s["reference_time"] + pd.to_timedelta(s["delay_used"], unit="D")
    s["log10_value"] = np.log10(s["value"])
    samples = s[["pathogen", "state", "series", "geo_value", "sample_index", "reference_time",
                 "week_end", "value", "log10_value", "delay_used", "available_date",
                 "population_served"]]
    samples.to_csv(os.path.join(OUT_DIR, "ww_samples.csv.gz"), index=False)

    # ---------- Steps 3-6: pseudo-real-time state values per forecast date ----------
    base = tuning_baselines(samples)
    last_origin = samples["week_end"].max() + pd.Timedelta(days=6)
    origins = fridays(FIRST_ORIGIN, last_origin)
    print(f"Building state values for {len(origins)} forecast dates ...")
    asof = pd.concat([state_values_asof(samples, o, base) for o in origins], ignore_index=True)
    asof.to_csv(os.path.join(OUT_DIR, "ww_state_asof.csv.gz"), index=False)

    # ---------- Step 7a (S1): true real-time from the version history ----------
    print("Building true real-time state values from versions (sensitivity S1) ...")
    arc = arc[arc["signal"].isin([f"{p}_avg_conc_lin" for p in PATHOGENS])]
    arc = arc.merge(site_state, on="geo_value", how="left")
    arc = arc.merge(kept_series, on=["pathogen", "series"], how="inner")
    arc = arc.merge(kept_states, on=["pathogen", "state"], how="inner")
    arc = arc[arc["value"] > 0]
    arc = arc.merge(h5_flag, on=["geo_value", "week_end"], how="left")
    arc = arc[~(arc["h5_flag"].fillna(False).astype(bool) & (arc["pathogen"] == "flu"))]
    arc["log10_value"] = np.log10(arc["value"])
    arc = arc.sort_values("report_date")

    rt_tables = []
    for o in fridays(FIRST_REALTIME_ORIGIN, arc["report_date"].max()):
        # latest version of each sample released by the forecast date
        known = (arc[arc["report_date"] <= o]
                 .drop_duplicates(["pathogen", "series", "sample_index"], keep="last")
                 .assign(available_date=o))
        rt_tables.append(state_values_asof(known, o, base))
    asof_rt = pd.concat(rt_tables, ignore_index=True)
    asof_rt.to_csv(os.path.join(OUT_DIR, "ww_state_asof_realtime.csv.gz"), index=False)

    # ---------- Step 7b (S2): revision size, first release vs latest ----------
    first = arc.drop_duplicates(["pathogen", "series", "sample_index"], keep="first")
    last = arc.drop_duplicates(["pathogen", "series", "sample_index"], keep="last")
    rev = first.merge(last, on=["pathogen", "series", "sample_index"], suffixes=("_first", "_last"))
    rev["log10_change"] = rev["log10_value_last"] - rev["log10_value_first"]
    rev_sum = (rev.groupby("pathogen")["log10_change"]
               .agg(samples="size",
                    share_revised=lambda x: (x.abs() > 1e-9).mean(),
                    median_abs_change=lambda x: x.abs().median(),
                    p90_abs_change=lambda x: x.abs().quantile(0.9))
               .reset_index())
    rev_sum.to_csv(os.path.join(OUT_DIR, "ww_revisions.csv"), index=False)

    # ---------- Summary ----------
    print("\n=========== Reporting delay (days from collection to first release) ===========")
    print("Samples used:", len(delay_samples))
    print(delay_samples.groupby("pathogen")["delay_days"].describe()[["count", "25%", "50%", "75%", "max"]])
    print("\nSeries using their own delay vs the pathogen median:")
    print(delays.groupby(["pathogen", "delay_source"]).size().unstack(fill_value=0))

    print(f"\nFlu samples dropped by the H5 rule: {n_h5}")
    print("\n=========== Clean samples ===========")
    print(samples.groupby("pathogen").agg(samples=("value", "size"),
                                          series=("series", "nunique"),
                                          states=("state", "nunique"),
                                          first_week=("week_end", "min"),
                                          last_week=("week_end", "max")))

    print("\n=========== State values per forecast date (pseudo-real-time) ===========")
    latest_week = asof.sort_values("week_end").groupby(["origin", "pathogen", "state"]).tail(1)
    lag = (latest_week["origin"] - latest_week["week_end"]).dt.days
    print("Forecast dates:", asof["origin"].nunique(),
          "| first:", asof["origin"].min().date(), "| last:", asof["origin"].max().date())
    print("Days from latest available wastewater week to the forecast date:")
    print(lag.describe()[["25%", "50%", "75%", "max"]])
    print("\nStates with a value in the latest week before each forecast date (median over dates):")
    fresh = latest_week[lag <= 6].groupby(["origin", "pathogen"]).size().groupby("pathogen").median()
    print(fresh)

    print("\n=========== Revisions (sensitivity S2) ===========")
    print(rev_sum.to_string(index=False))
    print("\nTrue real-time forecast dates (S1):", asof_rt["origin"].nunique())

    print("\nFiles saved in data/processed/. Copy everything above and send it back.")


if __name__ == "__main__":
    main()
