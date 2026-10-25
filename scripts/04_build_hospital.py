# 04_build_hospital.py
# =============================================================================
# Purpose: prepare the hospital-admission side of the study.
#
#   A. Versions table: every published version of weekly state admissions,
#      kept compactly (a row only when a value first appears or changes).
#      The model script uses it to rebuild what was known on any Friday.
#   B. Truth table: the latest version, used to score forecasts. Also fixes
#      the end of the evaluation period: the last target week must be at
#      least 4 weeks older than the truth snapshot, so late revisions settle.
#   C. Phases (Analysis Plan, section 5): for every state, pathogen and week,
#      the smoothed curve, weekly growth, season, turning points and phase.
#      Phases are computed from truth data and used ONLY to group results.
#
# Targets: flu, RSV, COVID-19 admissions, and RSV in adults 65+
#          (= RSV 65-74 + RSV 75+, the secondary target).
#
# Reads : data/raw/nhsn_archive/
# Writes: data/processed/nhsn_versions.csv.gz
#         data/processed/nhsn_truth.csv.gz
#         data/processed/phases.csv.gz
#         data/processed/evaluation_window.csv
# Run   : python scripts/04_build_hospital.py
# =============================================================================

import glob
import os

import numpy as np
import pandas as pd
from epiweeks import Week

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")

STATES = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo",
    "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa",
    "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
]
SIGNAL_TO_TARGET = {
    "confirmed_admissions_flu_ew": "flu",
    "confirmed_admissions_rsv_ew": "rsv",
    "confirmed_admissions_covid_ew": "covid",
    "confirmed_admissions_rsv_65_74_ew": "rsv_65_74",
    "confirmed_admissions_rsv_75_ew": "rsv_75",
}

FIRST_EVAL_ORIGIN = pd.Timestamp("2024-11-22")   # first Friday after versions begin
SETTLE_WEEKS = 4                                  # truth must be >= 4 weeks old
MAX_HORIZON = 4

# Phase settings (section 5)
SMOOTH_WEEKS = 3          # centered 3-week mean
TP_HALF_WINDOW = 4        # turning point = highest value within +/- 4 weeks
TP_MIN_SHARE = 0.25       # ... and at least 25% of the season peak
LOW_SHARE = 0.10          # low activity = below 10% of the season peak
PEAK_HALF_WINDOW = 1      # peak phase = within +/- 1 week of a turning point
GROWTH_BAND = 0.05        # growth if r > 0.05, decline if r < -0.05


def read_folder(name):
    files = sorted(glob.glob(os.path.join(RAW_DIR, name, f"{name}_*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"No files in {RAW_DIR}/{name}. Run 01_download_data.py first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def season_of(week_end):
    """Season label: a season runs from epiweek 31 to epiweek 30 of the next year.
    Example: week ending 2024-12-07 -> season '2024-25'."""
    w = Week.fromdate(week_end.date())
    start_year = w.year if w.week >= 31 else w.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def label_phases(df):
    """Section 5 steps for one (target, state) series, sorted by week.
    df has columns week_end, value (truth). Returns df with phase columns."""
    df = df.sort_values("week_end").copy()
    # 1. centered 3-week mean
    df["s"] = df["value"].rolling(SMOOTH_WEEKS, center=True, min_periods=SMOOTH_WEEKS).mean()
    # 2. weekly growth rate
    df["r"] = np.log(df["s"] + 1) - np.log(df["s"].shift(1) + 1)
    # 3. season and its peak
    df["season"] = df["week_end"].map(season_of)
    df["season_peak"] = df.groupby("season")["s"].transform("max")
    # 4. turning point: highest within +/- 4 weeks (full window only), above the
    #    previous week, and at least 25% of the season peak
    window = 2 * TP_HALF_WINDOW + 1
    local_max = df["s"].rolling(window, center=True, min_periods=window).max()
    df["turning_point"] = ((df["s"] == local_max) & (df["s"] > df["s"].shift(1))
                           & (df["s"] >= TP_MIN_SHARE * df["season_peak"]))
    # weeks within +/- 1 of a turning point
    tp = df["turning_point"].astype(int)
    near_tp = tp.rolling(2 * PEAK_HALF_WINDOW + 1, center=True, min_periods=1).max() > 0
    # 5. phase, rules applied in order
    df["phase"] = np.select(
        [df["s"].isna(),
         df["s"] < LOW_SHARE * df["season_peak"],
         near_tp,
         df["r"] > GROWTH_BAND,
         df["r"] < -GROWTH_BAND],
        ["undefined", "low", "peak", "growth", "decline"],
        default="plateau")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- read and tidy ----------------
    nh = read_folder("nhsn_archive")
    nh["state"] = nh["geo_value"].str.lower()
    nh = nh[nh["state"].isin(STATES) & nh["signal"].isin(SIGNAL_TO_TARGET)].copy()
    nh["target"] = nh["signal"].map(SIGNAL_TO_TARGET)
    nh["week_end"] = pd.to_datetime(nh["reference_time"])
    nh["report_date"] = (pd.to_datetime(nh["report_time"], utc=True)
                         .dt.tz_localize(None).dt.normalize())
    nh = nh.dropna(subset=["value"])

    # several reports on the same day: keep the last one
    nh = (nh.sort_values(["report_time"])
          .drop_duplicates(["target", "state", "week_end", "report_date"], keep="last"))

    # secondary target: RSV 65+ = 65-74 + 75+, only where both parts exist
    parts = nh[nh["target"].isin(["rsv_65_74", "rsv_75"])]
    wide = parts.pivot_table(index=["state", "week_end", "report_date"], columns="target",
                             values="value", aggfunc="first")
    # carry each part forward within its own version history so a version that
    # updates only one part still has both
    wide = wide.sort_index().groupby(level=["state", "week_end"]).ffill()
    rsv65 = wide.dropna().sum(axis=1).rename("value").reset_index().assign(target="rsv_65plus")
    main_targets = nh[nh["target"].isin(["flu", "rsv", "covid"])]
    allv = pd.concat([main_targets[["target", "state", "week_end", "report_date", "value"]],
                      rsv65[["target", "state", "week_end", "report_date", "value"]]],
                     ignore_index=True)

    # ---------------- A. compact versions ----------------
    allv = allv.sort_values(["target", "state", "week_end", "report_date"])
    prev = allv.groupby(["target", "state", "week_end"])["value"].shift(1)
    versions = allv[prev.isna() | (allv["value"] != prev)]
    versions.to_csv(os.path.join(OUT_DIR, "nhsn_versions.csv.gz"), index=False)

    # ---------------- B. truth and evaluation window ----------------
    truth = allv.groupby(["target", "state", "week_end"], as_index=False).last()
    snapshot = truth["report_date"].max()
    truth = truth.rename(columns={"report_date": "truth_report_date"})
    truth.to_csv(os.path.join(OUT_DIR, "nhsn_truth.csv.gz"), index=False)

    # last target week that has had SETTLE_WEEKS to settle (a Saturday)
    last_target = snapshot - pd.Timedelta(weeks=SETTLE_WEEKS)
    last_target = last_target - pd.Timedelta(days=(last_target.day_of_week - 5) % 7)
    # origin's latest data week ends 6 days before the origin; horizon-4 target
    # = that week + 4 weeks -> last origin = last_target - 4 weeks + 6 days
    last_origin = last_target - pd.Timedelta(weeks=MAX_HORIZON) + pd.Timedelta(days=6)
    window = pd.DataFrame({"truth_snapshot": [snapshot.date()],
                           "first_eval_origin": [FIRST_EVAL_ORIGIN.date()],
                           "last_eval_origin": [last_origin.date()],
                           "last_target_week": [last_target.date()],
                           "n_eval_origins": [len(pd.date_range(FIRST_EVAL_ORIGIN, last_origin, freq="W-FRI"))]})
    window.to_csv(os.path.join(OUT_DIR, "evaluation_window.csv"), index=False)

    # ---------------- C. phases ----------------
    # one (target, state) series at a time
    phases = pd.concat([label_phases(g) for _, g in truth.groupby(["target", "state"])],
                       ignore_index=True)
    phases = phases[["target", "state", "week_end", "value", "s", "r", "season",
                     "season_peak", "turning_point", "phase"]]
    phases.to_csv(os.path.join(OUT_DIR, "phases.csv.gz"), index=False)

    # ---------------- Summary ----------------
    print("\n=========== Evaluation window ===========")
    print(window.T.to_string(header=False))

    print("\n=========== Hospital versions ===========")
    v_sum = versions.groupby("target").agg(states=("state", "nunique"),
                                           weeks=("week_end", "nunique"),
                                           version_rows=("value", "size"),
                                           first_report=("report_date", "min"),
                                           last_report=("report_date", "max"))
    print(v_sum.to_string())

    # how much does a week's count change between first release and truth?
    first_rel = (versions.groupby(["target", "state", "week_end"], as_index=False).first()
                 .merge(truth[["target", "state", "week_end", "value"]],
                        on=["target", "state", "week_end"], suffixes=("_first", "_truth")))
    first_rel = first_rel[(first_rel["report_date"] >= FIRST_EVAL_ORIGIN)
                          & (first_rel["week_end"] <= last_target)
                          & (first_rel["value_truth"] >= 10)]
    first_rel["pct_change"] = 100 * (first_rel["value_truth"] - first_rel["value_first"]) / first_rel["value_truth"]
    print("\nChange from first release to truth, evaluation weeks with >= 10 admissions (%):")
    print(first_rel.groupby("target")["pct_change"].describe()[["count", "25%", "50%", "75%"]].round(1))

    # data lag on a Friday: latest week in the version released by that Friday
    lag = []
    for o in pd.date_range(FIRST_EVAL_ORIGIN, last_origin, freq="W-FRI"):
        known = versions[(versions["report_date"] <= o) & (versions["target"] == "flu")]
        lag.append((o - known["week_end"].max()).days)
    print("\nDays from latest published hospital week to each Friday (flu):",
          pd.Series(lag).value_counts().sort_index().to_dict())

    print("\n=========== Phases of evaluation target weeks ===========")
    ev = phases[phases["week_end"].between(FIRST_EVAL_ORIGIN, last_target)]
    print(pd.crosstab(ev["target"], ev["phase"]))
    print("\nTurning points in the evaluation weeks:")
    print(ev[ev["turning_point"]].groupby("target").size().to_string())

    print("\n=========== National check: peak week per season (sum over states) ===========")
    nat = (truth.groupby(["target", "week_end"], as_index=False)["value"].sum())
    nat["season"] = nat["week_end"].map(season_of)
    peaks = nat.loc[nat.groupby(["target", "season"])["value"].idxmax()]
    print(peaks[peaks["target"].isin(["flu", "rsv", "covid"])]
          .sort_values(["target", "season"]).to_string(index=False))

    # ---------------- wastewater freshness (promised check) ----------------
    ww_path = os.path.join(OUT_DIR, "ww_state_asof.csv.gz")
    if os.path.exists(ww_path):
        ww = pd.read_csv(ww_path, parse_dates=["origin", "week_end"])
        ww = ww[ww["origin"] >= FIRST_EVAL_ORIGIN]
        ww["weeks_back"] = ((ww["origin"] - ww["week_end"]).dt.days - 6) // 7
        print("\n=========== Wastewater: series per state by how old the week is ===========")
        print("(0 = the week that ended 6 days before the forecast date)")
        print(ww[ww["weeks_back"] <= 4].pivot_table(index="pathogen", columns="weeks_back",
                                                    values="n_series", aggfunc="median"))

    print("\nFiles saved in data/processed/. Copy everything above and send it back.")


if __name__ == "__main__":
    main()
