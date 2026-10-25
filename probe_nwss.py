# probe_nwss.py
# Purpose: before downloading everything, look at a SMALL piece of the wastewater
# data so we know exactly what we are dealing with. It answers three questions:
#   Q1. What site details ("auxiliary data") does Delphi give for each sewershed?
#       (we need: state, population served)
#   Q2. How do the four versions of the flu signal differ for one real site?
#       (avg_conc, avg_conc_lin, flowpop_lin, mic_lin)
#   Q3. What does the H5 bird-flu signal look like for that site?
# Everything printed is also saved as CSV files in the folder probe_outputs.
# Run with:  python probe_nwss.py

import os                                    # for making a folder
import pandas as pd                          # tables
from epidatpy import EpiDataContext, EpiRange

pd.set_option("display.width", 200)          # print wide tables on one line
pd.set_option("display.max_columns", 30)

os.makedirs("probe_outputs", exist_ok=True)  # folder for the CSV copies
epi = EpiDataContext()                       # connection to Delphi's API


def to_table(result):
    """Delphi sometimes returns a 'call' object and sometimes a table.
    This helper always gives back a pandas table."""
    return result.df() if hasattr(result, "df") else result


# ---------------------------------------------------------------------------
# Q1. Site details (auxiliary data)
# ---------------------------------------------------------------------------
print("\n================ Q1: site details (auxiliary data) ================")
try:
    aux = to_table(epi.epidata_aux("nwss", limit=200))   # only 200 rows, a sample
    print("Columns:", list(aux.columns))
    print("Number of rows in sample:", len(aux))
    print("\nFirst 3 rows, shown sideways so every column is visible:")
    print(aux.head(3).T)
    aux.to_csv("probe_outputs/q1_aux_sample.csv", index=False)
except Exception as e:                                   # keep going if this step fails
    print("Q1 FAILED:", repr(e))

# ---------------------------------------------------------------------------
# Q2. Pick one real site, then compare the four flu signal versions there
# ---------------------------------------------------------------------------
print("\n================ Q2: four versions of the flu signal ================")
site = None
try:
    # small pull just to find a site that reported flu in January 2026
    first = to_table(epi.epidata_snapshot(
        source="nwss",
        signals="flu_flowpop_lin",
        geo_type="sewershed",
        reference_time=EpiRange("2026-01-01", "2026-01-31"),
        limit=50,
    ))
    site = first["geo_value"].iloc[0]
    print("Site used for the comparison:", site)

    flu4 = to_table(epi.epidata_snapshot(
        source="nwss",
        signals=["flu_avg_conc", "flu_avg_conc_lin", "flu_flowpop_lin", "flu_mic_lin"],
        geo_type="sewershed",
        geo_values=site,
        reference_time=EpiRange("2025-10-01", "2026-03-31"),
    ))
    print("Columns:", list(flu4.columns))
    flu4.to_csv("probe_outputs/q2_flu_four_signals.csv", index=False)

    # one row per sample date, one column per signal, so the four can be compared side by side
    wide = flu4.pivot_table(index="reference_time", columns="signal",
                            values="value", aggfunc="first")
    print("\nSide by side (first 12 sample dates):")
    print(wide.head(12))
    print("\nSummary of each signal:")
    print(wide.describe().T)

    # extra columns that come with each value (sample index, PCR target, data source)
    for col in ["nwss_source", "pcr_target", "sample_index", "fill_method"]:
        if col in flu4.columns:
            print(f"\nValues seen in column '{col}':", flu4[col].dropna().unique()[:10])

    # Q2b. attach the site details to these rows (tests the merge we will use later)
    try:
        merged = to_table(epi.epidata_aux(flu4))
        new_cols = [c for c in merged.columns if c not in flu4.columns]
        print("\nSite-detail columns added by the merge:", new_cols)
        print(merged[new_cols].drop_duplicates().head(3).T)
        merged.to_csv("probe_outputs/q2b_flu_with_site_details.csv", index=False)
    except Exception as e:
        print("Q2b (merge) FAILED:", repr(e))
except Exception as e:
    print("Q2 FAILED:", repr(e))

# ---------------------------------------------------------------------------
# Q3. H5 bird-flu signal at the same site
# ---------------------------------------------------------------------------
print("\n================ Q3: H5 signal at the same site ================")
if site is not None:
    try:
        h5 = to_table(epi.epidata_snapshot(
            source="nwss",
            signals=["flu_h5_avg_conc", "flu_h5_flowpop_lin"],
            geo_type="sewershed",
            geo_values=site,
            reference_time=EpiRange("2025-10-01", "2026-03-31"),
        ))
        print("Rows:", len(h5))
        print(h5.head(10))
        h5.to_csv("probe_outputs/q3_h5.csv", index=False)
    except Exception as e:
        print("Q3 FAILED:", repr(e))
else:
    print("Skipped, because Q2 did not find a site.")

print("\nProbe finished. Copy everything above and send it back.")
