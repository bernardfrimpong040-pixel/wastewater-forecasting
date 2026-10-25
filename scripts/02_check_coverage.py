# 02_check_coverage.py
# =============================================================================
# Purpose: using ONLY data availability (no forecasting, no scoring), check
# which state-pathogen pairs meet the inclusion rule in the Analysis Plan:
#
#   Rule (section 2): a state-pathogen pair is included if wastewater was
#   reported in at least 75% of tuning-period weeks.
#   Fallback (section 8): if too many states fail, use 75% of the first
#   8 evaluation-period weeks instead.
#   Detection-limit rule (section 3): a series (site x lab) is excluded if
#   more than half of its baseline samples are below the detection limit.
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
LOD_SHARE_LIMIT = 0.50         # exclude series with >50% below detection limit

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
    return dates + pd.to_timedelta((5 - dates.dt.weekday) % 7, unit="D")


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # ============================ WASTEWATER ============================
    ww = read_folder("nwss_latest")
    ww["week_end"] = week_ending_saturday(ww["reference_time"])
    ww["pathogen"] = ww["signal"].str.split("_").str[0]          # flu / rsv / covid
    ww["state"] = ww["state_territory"].str.lower()

    # A series is one site tested by one lab
    ww["series"] = ww["geo_value"] + "|" + ww["nwss_source"]

    # --- Detection-limit rule (per series, over its baseline samples) ---
    # Raw concentration rows (signal ends in "_avg_conc") tell us if a sample
    # was below the lab's detection limit (lod_sewage).
    raw = ww[ww["signal"].isin([f"{p}_avg_conc" for p in PATHOGENS])].copy()
    raw["lod_sewage"] = pd.to_numeric(raw["lod_sewage"], errors="coerce")
    raw["below_lod"] = raw["value"] < raw["lod_sewage"]

    # Baseline = tuning period; series that start later use their first 8 weeks
    in_tuning = raw["week_end"].between(TUNING_FIRST_WEEK_END, TUNING_LAST_WEEK_END)
    first_week = raw.groupby("series")["week_end"].transform("min")
    in_first8 = raw["week_end"] < first_week + pd.Timedelta(weeks=8)
    has_tuning = raw.groupby("series")["week_end"].transform(
        lambda w: w.between(TUNING_FIRST_WEEK_END, TUNING_LAST_WEEK_END).any())
    raw["in_baseline"] = (has_tuning & in_tuning) | (~has_tuning & in_first8)

    series_lod = (raw[raw["in_baseline"]]
                  .groupby(["pathogen", "state", "series"])
                  .agg(baseline_samples=("below_lod", "size"),
                       share_below_lod=("below_lod", "mean"))
                  .reset_index())
    series_lod["excluded_lod"] = series_lod["share_below_lod"] > LOD_SHARE_LIMIT
    series_lod.to_csv(os.path.join(SUMMARY_DIR, "series_lod.csv"), index=False)

    print("\n=========== Series (site x lab) and the detection-limit rule ===========")
    lod_table = (series_lod.groupby("pathogen")
                 .agg(series=("series", "size"),
                      excluded_by_lod_rule=("excluded_lod", "sum"),
                      median_share_below_lod=("share_below_lod", "median")))
    print(lod_table)

    # --- Weekly coverage per state, using only series that pass the LOD rule ---
    keep_series = set(series_lod.loc[~series_lod["excluded_lod"], "series"])
    primary = ww[ww["signal"].isin([f"{p}_avg_conc_lin" for p in PATHOGENS])
                 & ww["series"].isin(keep_series)]
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

    # number of usable series per state (latest count) and population served
    n_series = (primary.groupby(["pathogen", "state"])["series"].nunique()
                .rename("usable_series"))
    cov = cov.merge(n_series.reset_index(), on=["pathogen", "state"], how="left")
    cov["usable_series"] = cov["usable_series"].fillna(0).astype(int)
    cov.to_csv(os.path.join(SUMMARY_DIR, "coverage_by_state.csv"), index=False)

    print("\n=========== Inclusion rule: states passing, per pathogen (of 51) ===========")
    print(cov.groupby("pathogen")[["passes_main_rule", "passes_fallback_rule"]].sum())

    print("\n=========== States FAILING the main rule (share of tuning weeks) ===========")
    for p in PATHOGENS:
        failing = cov[(cov["pathogen"] == p) & ~cov["passes_main_rule"]]
        listing = ", ".join(f"{s.upper()} {v:.0%}" for s, v in
                            zip(failing["state"], failing["share_tuning"]))
        print(f"{p}: {len(failing)} states -> {listing if listing else 'none'}")

    # ============================ HOSPITALS ============================
    nhsn = read_folder("nhsn_archive")
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
