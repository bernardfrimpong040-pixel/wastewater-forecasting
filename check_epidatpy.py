# check_epidatpy.py
# Purpose: test that Delphi's official Python client (epidatpy) works on this laptop.
# It does two things:
#   1. asks Delphi which wastewater (nwss) and hospital (nhsn) signals exist
#   2. downloads one small real piece of data: Ohio flu hospital admissions
#      AS THEY WERE KNOWN on 10 January 2025 (a "snapshot" / old version)
# Run it from the terminal with:  python check_epidatpy.py

from epidatpy import EpiDataContext   # Delphi's official Python client

# Step 0: open a connection to Delphi's API
epi = EpiDataContext()

# Step 1: metadata for the two sources we need
for source in ["nwss", "nhsn"]:
    meta = epi.epidata_meta(source=source)          # a dictionary describing the source
    print("\n=== Source:", source, "===")
    print("Number of signals :", len(meta["signals"]))
    print("Geographic levels :", meta["geo_types"])
    print("Reference dates   :", meta["reference_time_range"])   # which weeks the data cover
    print("Version dates     :", meta["report_time_range"])      # how far back old versions go

# Step 2: one small real download, as known on 10 January 2025
call = epi.epidata_snapshot(
    source="nhsn",                            # hospital data
    signals="confirmed_admissions_flu_ew",    # weekly flu admissions
    geo_type="state",
    geo_values="oh",                          # Ohio
    snapshot_date="2025-01-10",               # what Delphi knew on this date
)
df = call.df()                                # turn the result into a pandas table
print("\n=== Ohio flu admissions, as known on 2025-01-10 (last 5 weeks) ===")
print(df.tail(5))

print("\nTest finished. Copy everything above and send it back.")
