# Does wastewater surveillance improve hospital-admission forecasts?

A real-time, phase-stratified evaluation of whether adding wastewater data
(CDC National Wastewater Surveillance System, NWSS) improves 1–4 week ahead
probabilistic forecasts of weekly state hospital admissions (CDC NHSN) for
**influenza, RSV and COVID-19** across US states, 2024–2026.

**Author:** Bernard Frimpong  
**Status:** in progress — data preparation stage (Week 2 of 8)

## Research question

For each pathogen, how much does adding wastewater to a forecasting model
change the weighted interval score (WIS) of its forecasts? Is any gain
concentrated in growth and peak weeks (epidemic turning points)?

Each model is run twice, identical except for the wastewater inputs, so the
difference between the two runs is the effect of wastewater. All analysis
choices were fixed in a written analysis plan before any forecast was scored.

## Data

All data are public and downloaded through
[Delphi's Epidata API](https://cmu-delphi.github.io/delphi-epidata/) using the
official Python client `epidatpy`:

| Data | Delphi source | Versions ("as known on date X") from |
|---|---|---|
| Weekly hospital admissions by state (flu, RSV, COVID-19; RSV 65+) | `nhsn` | 19 Nov 2024 |
| Wastewater concentrations by sewershed (flu A, RSV, SARS-CoV-2, H5) plus site details | `nwss` | 25 Feb 2026 |

Versioned data let each forecast use only what was published on its forecast
date, which prevents information from the future leaking into the results.

## Project structure

```
scripts/
  01_download_data.py    download all data into data/raw/ (resumable)
  02_check_coverage.py   inclusion check: which states have enough wastewater
  03_build_wastewater.py weekly state wastewater values as known on each
                         Friday forecast date; reporting delays; revisions
data/                    created by the scripts; not tracked by git
  raw/                   monthly compressed CSV downloads
  summary/               coverage tables
  processed/             cleaned inputs for the forecasting models
```

## How to run

Requires Python 3.9 or newer (developed on Python 3.13, Windows).

1. Get a free Delphi API key at
   <https://api.delphi.cmu.edu/epidata/admin/registration_form> and save it as
   an environment variable (Windows PowerShell):
   `setx DELPHI_EPIDATA_KEY "your-key"`, then reopen PowerShell.
   The key is never stored in this repository.
2. Create the environment and install packages:
   ```
   py -3.13 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install epidatpy pandas
   ```
3. Run the scripts in order:
   ```
   python scripts/01_download_data.py
   python scripts/02_check_coverage.py
   python scripts/03_build_wastewater.py
   ```

## Licence and data use

Code: to be released with the paper under an open licence, archived on Zenodo
with a DOI. Data: US government public data (NHSN, NWSS) accessed through the
Delphi Epidata API; please cite Delphi and the original CDC sources.
