# 02_check_coverage.py
# =============================================================================
# Purpose: using ONLY data availability (no forecasting, no scoring), check
# which state-pathogen pairs meet the inclusion rule in the Analysis Plan:
#
#   Rule (section 2): a state-pathogen pair is included if wastewater was
#   reported in at least 75% of tuning-period weeks.
#   Fallback (section 8): if too many states fail, use 75% of the first
#   8 evaluation-period weeks instead.
#   Detection-limit rule (section 3, corrected 24 Oct 2026): a series (site x
#   lab) is excluded for a pathogen if more than half of its baseline samples
#   taken IN SEASON are below the detection limit. "In season" = weeks when the
#   state's admissions for that pathogen are at least 10% of the state's
#   2023-24 season peak. (Flu and RSV are genuinely absent in summer, so
#   off-season zeros are real and must not count against a series.)
#
# It also checks the hospital data: states, weeks and number of versions.
#
# Reads : data/raw/nwss_latest/*.csv.gz and data/raw/nhsn_archive/*.csv.gz
# Writes: data/summary/coverage_by_state.csv and data/summary/series_lod.csv
# Run   : python scripts/02_check_coverage.py
# =============================================================================

import glob                    # find all monthly files
import os

import pandas as pd

RAW_DIR = os.path.join("data", "raw")
SUMMARY_DIR = os.path.join("data", "summary")

# Tuning period: from Oct 2023 up to the last Saturday before the first
# forecast origin (Friday 22 Nov 2024), i.e. weeks ending 7 Oct 2023 .. 16 Nov 2024.
TUNING_FIRST_WEEK_END = pd.Timestamp("2023-10-07")
TUNING_LAST_WEEK_END = pd.Timestamp("2024-11-16")

# Fallback window: the first 8 evaluation weeks (week ending 23 Nov 2024 .. 11 Jan 2025)
FALLBACK_FIRST_WEEK_END = pd.Timestamp("2024-11-23")
FALLBACK_LAST_WEEK_END = pd.Timestamp("2025-01-11")

THRESHOLD = 0.75               # 75% of weeks
LOD_SHARE_LIMIT = 0.50         # exclude series with >50% of in-season samples below detection limit

# In-season definition (same 10% threshold as the low-activity phase, section 5)
SEASON_2324_FIRST_WEEK_END = pd.Timestamp("2023-08-05")   # epiweek 31 of 2023
SEASON_2324_LAST_WEEK_END = pd.Timestamp("2024-07-27")    # epiweek 30 of 2024
IN_SEASON_SHARE = 0.10         # in season = admissions >= 10% of the 2023-24 peak
MIN_STATE_PEAK = 20            # if a state's 2023-24 peak is below this (sparse or
                               # voluntary reporting, e.g. RSV), use national weeks
MIN_INSEASON_SAMPLES = 4       # fewer in-season baseline samples -> rule cannot be
                               # judged, series is kept

# Hospital signal for each pathogen (the age-group RSV signals are not used here)
NHSN_SIGNAL = {"flu": "confirmed_admissions_flu_ew",
               "rsv": "confirmed_admissions_rsv_ew",
               "covid": "confirmed_admissions_covid_ew"}

# The 50 states and DC (Delphi uses lower-case two-letter codes)
STATES = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo",
    "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa",
    "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
]
PATHOGENS = ["flu", "rsv", "covid"]


def read_folder(name):
    """Read and stack every monthly file of one dataset."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, name, f"{name}_*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"No files in {RAW_DIR}/{name}. Run 01_download_data.py first.")
    tables = [pd.read_csv(f, dtype={"geo_value": str, "sample_index": str,
                                    "county_fips": str}) for f in files]
    tables = [t for t in tables if len(t) > 0]
    return pd.concat(tables, ignore_index=True)


def week_ending_saturday(dates):
    """MMWR weeks run Sunday-Saturday; return the Saturday that ends each date's week.
    Monday=0 ... Saturday=5, Sunday=6, so days to add = (5 - weekday) mod 7."""
    dates = pd.to_datetime(dates)
    return dates + pd.to_timedelta((5 - dates.dt.day_of_week) % 7, unit="D")


def in_season_weeks(nhsn):
    """Return a table of (pathogen, state, week_end, in_season).

    Uses the LATEST version of each weekly admission count. For each state and
    pathogen, a week is in season if admissions are at least 10% of that state's
    2023-24 season peak. States whose 2023-24 peak is below MIN_STATE_PEAK use
    the national in-season weeks instead (sum of the 50 states and DC)."""
    nhsn = nhsn[nhsn["signal"].isin(NHSN_SIGNAL.values())].copy()
    nhsn["pathogen"] = nhsn["signal"].map({v: k for k, v in NHSN_SIGNAL.items()})
    nhsn["state"] = nhsn["geo_value"].str.lower()
    nhsn = nhsn[nhsn["state"].isin(STATES)]
    nhsn["week_end"] = pd.to_datetime(nhsn["reference_time"])
    nhsn["report_time"] = pd.to_datetime(nhsn["report_time"], utc=True)

    # latest version of each (pathogen, state, week)
    latest = (nhsn.sort_values("report_time")
              .groupby(["pathogen", "state", "week_end"], as_index=False)
              .last()[["pathogen", "state", "week_end", "value"]])

    # national series = sum over the 50 states and DC
    national = (latest.groupby(["pathogen", "week_end"], as_index=False)["value"].sum()
                .assign(state="us"))
    both = pd.concat([latest, national], ignore_index=True)

    # 2023-24 season peak for each (pathogen, state)
    in_2324 = both["week_end"].between(SEASON_2324_FIRST_WEEK_END, SEASON_2324_LAST_WEEK_END)
    peaks = (both[in_2324].groupby(["pathogen", "state"])["value"].max()
             .rename("peak_2324").reset_index())
    both = both.merge(peaks, on=["pathogen", "state"], how="left")
    both["in_season"] = both["value"] >= IN_SEASON_SHARE * both["peak_2324"]

    # states with a missing or tiny 2023-24 peak borrow the national in-season weeks
    nat = (both[both["state"] == "us"][["pathogen", "week_end", "in_season"]]
           .rename(columns={"in_season": "in_season_national"}))
    states = both[both["state"] != "us"].merge(nat, on=["pathogen", "week_end"], how="left")
    sparse = states["peak_2324"].isna() | (states["peak_2324"] < MIN_STATE_PEAK)
    states["used_national"] = sparse
    states.loc[sparse, "in_season"] = states.loc[sparse, "in_season_national"]
    return states[["pathogen", "state", "week_end", "in_season", "used_national"]]


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # ============================ HOSPITALS ============================
    # Read first: the in-season weeks come from hospital admissions.
    nhsn = read_folder("nhsn_archive")
    season = in_season_weeks(nhsn)

    # ============================ WASTEWATER ============================
    ww = read_folder("nwss_latest")
    ww["week_end"] = week_ending_saturday(ww["reference_time"])
    ww["pathogen"] = ww["signal"].str.split("_").str[0]          # flu / rsv / covid
    ww["state"] = ww["state_territory"].str.lower()

    # A series is one site tested by one lab. The same series can be kept for
    # one pathogen and excluded for another, so every rule below works on the
    # pair (pathogen, series), never on the series alone.
    ww["series"] = ww["geo_value"] + "|" + ww["nwss_source"]
    KEY = ["pathogen", "series"]

    # --- Detection-limit rule (per pathogen and series, in-season baseline samples) ---
    # Raw concentration rows (signal ends in "_avg_conc") tell us if a sample
    # was below the lab's detection limit (lod_sewage).
    raw = ww[ww["signal"].isin([f"{p}_avg_conc" for p in PATHOGENS])].copy()
    raw["lod_sewage"] = pd.to_numeric(raw["lod_sewage"], errors="coerce")
    raw["below_lod"] = raw["value"] < raw["lod_sewage"]

    # attach in-season flag for the sample's state and week
    raw = raw.merge(season, on=["pathogen", "state", "week_end"], how="left")
    raw["in_season"] = raw["in_season"].fillna(False).astype(bool)

    # Baseline = tuning period; series with no tuning data use their first 8
    # IN-SEASON weeks instead.
    in_tuning = raw["week_end"].between(TUNING_FIRST_WEEK_END, TUNING_LAST_WEEK_END)
    has_tuning = (raw.assign(t=in_tuning).groupby(KEY)["t"].transform("any"))
    ins = raw[raw["in_season"]]
    first_ins_week = ins.groupby(KEY)["week_end"].min().rename("first_ins_week")
    raw = raw.merge(first_ins_week.reset_index(), on=KEY, how="left")
    in_first8 = raw["week_end"] < raw["first_ins_week"] + pd.Timedelta(weeks=8)
    raw["in_baseline"] = raw["in_season"] & ((has_tuning & in_tuning) | (~has_tuning & in_first8))

    base = raw[raw["in_baseline"]]
    series_lod = (base.groupby(["pathogen", "state", "series"])
                  .agg(inseason_baseline_samples=("below_lod", "size"),
                       share_below_lod=("below_lod", "mean"))
                  .reset_index())
    # series with no or too few in-season baseline samples: rule cannot be judged -> kept
    all_series = raw[["pathogen", "state", "series"]].drop_duplicates()
    series_lod = all_series.merge(series_lod, on=["pathogen", "state", "series"], how="left")
    series_lod["inseason_baseline_samples"] = series_lod["inseason_baseline_samples"].fillna(0).astype(int)
    judged = series_lod["inseason_baseline_samples"] >= MIN_INSEASON_SAMPLES
    series_lod["excluded_lod"] = judged & (series_lod["share_below_lod"] > LOD_SHARE_LIMIT)
    series_lod["rule_judged"] = judged
    series_lod.to_csv(os.path.join(SUMMARY_DIR, "series_lod.csv"), index=False)

    print("\n=========== Series (site x lab) and the in-season detection-limit rule ===========")
    lod_table = (series_lod.groupby("pathogen")
                 .agg(series=("series", "size"),
                      judged=("rule_judged", "sum"),
                      excluded_by_lod_rule=("excluded_lod", "sum"),
                      median_share_below_lod=("share_below_lod", "median")))
    print(lod_table)

    # --- Weekly coverage per state, using only (pathogen, series) pairs that pass ---
    kept = series_lod.loc[~series_lod["excluded_lod"], KEY]
    primary = ww[ww["signal"].isin([f"{p}_avg_conc_lin" for p in PATHOGENS])]
    primary = primary.merge(kept, on=KEY, how="inner")
    state_weeks = primary[["pathogen", "state", "week_end"]].drop_duplicates()

    def share_of_weeks(first, last):
        """Share of weeks in [first, last] with at least one usable sample, per state-pathogen."""
        n_weeks = len(pd.date_range(first, last, freq="7D"))
        window = state_weeks[state_weeks["week_end"].between(first, last)]
        counts = window.groupby(["pathogen", "state"]).size()
        full = pd.MultiIndex.from_product([PATHOGENS, STATES], names=["pathogen", "state"])
        return (counts.reindex(full, fill_value=0) / n_weeks).rename("share")

    cov = pd.concat({
        "share_tuning": share_of_weeks(TUNING_FIRST_WEEK_END, TUNING_LAST_WEEK_END),
        "share_fallback": share_of_weeks(FALLBACK_FIRST_WEEK_END, FALLBACK_LAST_WEEK_END),
    }, axis=1).reset_index()
    cov["passes_main_rule"] = cov["share_tuning"] >= THRESHOLD
    cov["passes_fallback_rule"] = cov["share_fallback"] >= THRESHOLD

    # number of usable series per state, and whether national in-season weeks were used
    n_series = (primary.groupby(["pathogen", "state"])["series"].nunique()
                .rename("usable_series"))
    cov = cov.merge(n_series.reset_index(), on=["pathogen", "state"], how="left")
    cov["usable_series"] = cov["usable_series"].fillna(0).astype(int)
    nat_flag = season.groupby(["pathogen", "state"])["used_national"].any().reset_index()
    cov = cov.merge(nat_flag, on=["pathogen", "state"], how="left")
    cov.to_csv(os.path.join(SUMMARY_DIR, "coverage_by_state.csv"), index=False)

    print("\n=========== Inclusion rule: states passing, per pathogen (of 51) ===========")
    print(cov.groupby("pathogen")[["passes_main_rule", "passes_fallback_rule"]].sum())

    print("\n=========== Usable series per state (main-rule states) ===========")
    ok = cov[cov["passes_main_rule"]]
    print(ok.groupby("pathogen")["usable_series"].describe()[["min", "25%", "50%", "75%", "max"]])

    print("\n=========== States FAILING the main rule (share of tuning weeks) ===========")
    for p in PATHOGENS:
        failing = cov[(cov["pathogen"] == p) & ~cov["passes_main_rule"]]
        listing = ", ".join(f"{s.upper()} {v:.0%}" for s, v in
                            zip(failing["state"], failing["share_tuning"]))
        print(f"{p}: {len(failing)} states -> {listing if listing else 'none'}")

    print("\n=========== States using national in-season weeks (sparse 2023-24 data) ===========")
    for p in PATHOGENS:
        used = cov[(cov["pathogen"] == p) & cov["used_national"].fillna(False)]["state"]
        print(f"{p}: {', '.join(s.upper() for s in used) if len(used) else 'none'}")

    print("\n=========== Hospital data (NHSN) ===========")
    nhsn_sum = (nhsn.groupby("signal")
                .agg(states=("geo_value", "nunique"),
                     first_week=("reference_time", "min"),
                     last_week=("reference_time", "max"),
                     versions=("report_time", "nunique"),
                     rows=("value", "size")))
    print(nhsn_sum.to_string())
    missing_states = sorted(set(STATES) - set(nhsn["geo_value"].str.lower()))
    print("States missing from hospital data:", missing_states if missing_states else "none")

    print("\nFiles saved in data/summary/. Copy everything above and send it back.")


if __name__ == "__main__":
    main()
