# 01_download_data.py
# =============================================================================
# Purpose: download every dataset the study needs from Delphi's Epidata API
# and save it as compressed CSV files in data/raw/.
#
# What it downloads (see the Analysis Plan, sections 2 and 3):
#   A. nhsn_archive : weekly state hospital admissions (flu, RSV, COVID-19,
#                     RSV age 65-74 and 75+) WITH every weekly version since
#                     19 Nov 2024, so we can rebuild "what was known on date X".
#   B. nwss_latest  : wastewater measurements per sewershed, latest version,
#                     with site details (state, population, detection limit,
#                     PMMoV fecal marker) attached.
#   C. nwss_archive : wastewater measurements WITH every version since
#                     25 Feb 2026, to measure reporting delays and revisions.
#
# The download is split into one file per month of data. If the script stops
# (internet drops, laptop sleeps), just run it again: months already saved
# are skipped, so it continues where it left off.
#
# Before running: save your Delphi API key once with
#     setx DELPHI_EPIDATA_KEY "your-key"
# and reopen PowerShell. The key is NEVER written in this file.
#
# Run from the project folder with:   python scripts/01_download_data.py
# =============================================================================

import os                      # folders and file paths
import time                    # waiting between retries
import warnings                # to quiet a repeated, harmless warning
from datetime import date      # today's date

import pandas as pd            # tables
from epidatpy import EpiDataContext, EpiRange   # Delphi's official client

# -----------------------------------------------------------------------------
# Settings (change here, nowhere else)
# -----------------------------------------------------------------------------
RAW_DIR = os.path.join("data", "raw")   # where all downloads go

# A. Hospital admissions: start in Aug 2022 so the 2023-24 tuning season
#    has a full year of history before it.
NHSN_START = "2022-08-01"
NHSN_SIGNALS = [
    "confirmed_admissions_flu_ew",
    "confirmed_admissions_rsv_ew",
    "confirmed_admissions_covid_ew",
    "confirmed_admissions_rsv_65_74_ew",     # secondary target (adults 65+)
    "confirmed_admissions_rsv_75_ew",
]

# B. Wastewater, latest version: start in Jul 2023 (the 2023-24 tuning season)
NWSS_LATEST_START = "2023-07-01"
NWSS_LATEST_SIGNALS = [
    # primary signal: concentration, below-detection-limit samples = half the limit
    "flu_avg_conc_lin", "rsv_avg_conc_lin", "covid_avg_conc_lin",
    # raw concentration, used to find samples below the detection limit
    "flu_avg_conc", "rsv_avg_conc", "covid_avg_conc",
    # sensitivity analysis: copies per person per day
    "flu_flowpop_lin", "rsv_flowpop_lin", "covid_flowpop_lin",
    # H5 bird-flu rule
    "flu_h5_avg_conc",
]

# Site details we keep from the auxiliary data (the rest are lab-method details)
AUX_KEEP = [
    "state_territory", "county_fips", "population_served",
    "sample_matrix", "pcr_target_units", "lod_sewage",
    "hum_frac_target_mic", "hum_frac_mic_conc", "hum_frac_mic_unit",
    "flow_rate",
]

# C. Wastewater, all versions: versions exist from 25 Feb 2026, so samples
#    collected from Jan 2026 onward are enough to measure delays and revisions.
NWSS_ARCHIVE_START = "2026-01-01"
NWSS_ARCHIVE_SIGNALS = ["flu_avg_conc_lin", "rsv_avg_conc_lin", "covid_avg_conc_lin"]

TODAY = date.today()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def month_ranges(start_text, end_date):
    """List of (first day, last day) for every month from start_text to end_date.
    Example: month_ranges("2026-01-01", date(2026, 3, 10)) gives
    [("2026-01-01","2026-01-31"), ("2026-02-01","2026-02-28"), ("2026-03-01","2026-03-31")]"""
    ranges = []
    first = pd.Timestamp(start_text)
    while first.date() <= end_date:
        last = first + pd.offsets.MonthEnd(0)          # last day of that month
        ranges.append((first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")))
        first = last + pd.Timedelta(days=1)            # first day of next month
    return ranges


def to_table(result):
    """Delphi returns either a 'call' object or a table; always give back a table."""
    return result.df() if hasattr(result, "df") else result


def download_by_month(name, fetch_month, start_text, log):
    """Download one dataset month by month, saving each month to its own file.

    name        : dataset name, also the sub-folder name (e.g. "nhsn_archive")
    fetch_month : a function that takes (first_day, last_day) and returns a table
    start_text  : first month to download, e.g. "2022-08-01"
    log         : list where we record one line per month (for the summary)
    """
    folder = os.path.join(RAW_DIR, name)
    os.makedirs(folder, exist_ok=True)

    for first_day, last_day in month_ranges(start_text, TODAY):
        path = os.path.join(folder, f"{name}_{first_day[:7]}.csv.gz")

        # Recent months are still filling up and being revised, so any month
        # ending within the last 60 days is always downloaded fresh.
        is_recent = pd.Timestamp(last_day) >= pd.Timestamp(TODAY) - pd.Timedelta(days=60)
        if os.path.exists(path) and not is_recent:
            print(f"  {name} {first_day[:7]}: already saved, skipping")
            continue

        # Try up to 3 times; Delphi's servers occasionally time out.
        table = None
        for attempt in range(1, 4):
            try:
                table = fetch_month(first_day, last_day)
                break
            except Exception as e:
                print(f"  {name} {first_day[:7]}: attempt {attempt} failed ({e!r})")
                if attempt < 3:
                    time.sleep(30 * attempt)           # wait 30 s, then 60 s

        if table is None:
            print(f"  {name} {first_day[:7]}: FAILED 3 times. Run the script again later.")
            log.append((name, first_day[:7], "FAILED"))
            continue

        table.to_csv(path, index=False)                # .gz ending = compressed
        print(f"  {name} {first_day[:7]}: {len(table):,} rows saved")
        log.append((name, first_day[:7], len(table)))


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------
def main():
    # Warn (once) if the API key is missing: large downloads need it.
    if not os.environ.get("DELPHI_EPIDATA_KEY"):
        print("WARNING: DELPHI_EPIDATA_KEY is not set. Run setx (see top of file),")
        print("reopen PowerShell, and try again. Stopping so we do not hit rate limits.")
        return
    warnings.filterwarnings("ignore", message=".*DELPHI_EPIDATA_KEY.*")

    epi = EpiDataContext()           # connection to Delphi
    log = []                         # one entry per month per dataset

    # ---------------- A. hospital admissions, all versions ----------------
    print("\nA. Hospital admissions (NHSN), every weekly version")

    def fetch_nhsn(first_day, last_day):
        return to_table(epi.epidata_archive(
            source="nhsn",
            signals=NHSN_SIGNALS,
            geo_type="state",
            reference_time=EpiRange(first_day, last_day),
            return_empty=True,                         # empty table instead of error
        ))

    download_by_month("nhsn_archive", fetch_nhsn, NHSN_START, log)

    # ---------------- B. wastewater, latest version + site details ----------------
    print("\nB. Wastewater (NWSS), latest version, with site details")

    def fetch_nwss_latest(first_day, last_day):
        table = to_table(epi.epidata_snapshot(
            source="nwss",
            signals=NWSS_LATEST_SIGNALS,
            geo_type="sewershed",
            reference_time=EpiRange(first_day, last_day),
            return_empty=True,
        ))
        if len(table) == 0:
            return table
        # attach site details (state, population, detection limit, PMMoV)
        merged = to_table(epi.epidata_aux(table))
        keep = list(table.columns) + [c for c in AUX_KEEP if c in merged.columns]
        return merged[keep]

    download_by_month("nwss_latest", fetch_nwss_latest, NWSS_LATEST_START, log)

    # ---------------- C. wastewater, all versions since Feb 2026 ----------------
    print("\nC. Wastewater (NWSS), every version since 25 Feb 2026")

    def fetch_nwss_archive(first_day, last_day):
        return to_table(epi.epidata_archive(
            source="nwss",
            signals=NWSS_ARCHIVE_SIGNALS,
            geo_type="sewershed",
            reference_time=EpiRange(first_day, last_day),
            return_empty=True,
        ))

    download_by_month("nwss_archive", fetch_nwss_archive, NWSS_ARCHIVE_START, log)

    # ---------------- Summary ----------------
    print("\n================ DOWNLOAD SUMMARY ================")
    summary = pd.DataFrame(log, columns=["dataset", "month", "rows"])
    if len(summary) == 0:
        print("Nothing new was downloaded (everything already saved).")
    else:
        print(summary.to_string(index=False))
        failed = summary[summary["rows"] == "FAILED"]
        print(f"\nMonths failed: {len(failed)}  (run the script again to retry them)")
    print("\nNext: run scripts/02_check_coverage.py and send me its output.")


if __name__ == "__main__":
    main()
