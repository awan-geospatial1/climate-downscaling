# Climate Downscaling & Bias Correction

An end-to-end pipeline that pulls CMIP6 climate projections and observational reference
data from **Google Earth Engine**, bias-corrects the projections against the reference
using **Quantile Delta Mapping (QDM)**, computes standard temperature/precipitation
climate indices, and exports an Excel summary plus fan charts and spatial maps — all
for a single area of interest (AOI) defined by a shapefile.

Runs interactively in Google Colab via `run_interactive.ipynb`, or programmatically via
`main.run_pipeline(params)`.

## What it does

1. **Reference data** — fetches daily precipitation (CHIRPS) and temperature
   (ERA5-Land) for the baseline period over your AOI.
2. **CMIP6 historical** — fetches matching historical GCM output (NASA/GDDP-CMIP6),
   regrids it to the reference grid, and trains a QDM bias-correction model per
   variable/model.
3. **CMIP6 future** — fetches future SSP scenario data for each requested time
   window, applies the trained QDM correction, and (for precipitation) adjusts wet-day
   frequency to match the reference.
4. **Indices** — computes annual/monthly temperature and precipitation indices
   (means, threshold-exceedance days, wet/dry season totals, Rx1day, GEV return
   levels, etc.) for the baseline and every scenario/period/model, then aggregates
   across models (mean, p10, p90).
5. **Outputs** — an Excel summary table, fan charts (historical + scenario spread
   over time) for temperature and precipitation, and spatial maps of ensemble-mean
   indices clipped to your AOI.

## Repository contents

| File | Purpose |
|---|---|
| `config.py` | Per-variable settings: GEE source collection/band, units, QDM kind, clip ranges, wet-day adjustment flag, unit conversion factor. |
| `gee_utils.py` | Fetches reference and CMIP6 data from Earth Engine via `xee`, chunked by 5-year windows; cleans/regrids to a common grid. |
| `qdm_utils.py` | Trains/applies QDM bias correction (`xsdba`); adjusts precipitation wet-day frequency. |
| `indices_utils.py` | Computes climate indices with `xclim`; GEV return-level bootstrapping for extreme precipitation. |
| `plot_utils.py` | Fan charts and spatial maps (via `rioxarray`/`matplotlib`). |
| `main.py` | Orchestrates the full pipeline (`run_pipeline`), builds the Excel summary. |
| `run_interactive.ipynb` | Colab notebook with `ipywidgets` UI for shapefile upload and parameter entry. |
| `requirements.txt` | Pinned dependencies. |

## Quick start (Google Colab)

1. Open `run_interactive.ipynb` in Colab.
2. Run the first cell — it mounts Drive, clones this repo, and installs
   `requirements.txt`.
3. Upload a shapefile (`.zip`, `.shp`, `.gpkg`, or `.geojson`) or point to one on
   Drive, fill in the GCP project ID and parameters, and click **Run Pipeline**.

Or manually, in any Colab cell:

```python
!git clone https://github.com/awan-geospatial1/climate-downscaling.git
%cd climate-downscaling
!pip install -q -r requirements.txt --upgrade
```
**Restart the runtime after installing** (Colab preloads its own older versions of
`xarray`/`shapely`/`pyproj`, and those can shadow the freshly installed ones until a
restart).

Then:

```python
from main import run_pipeline

params = {
    'shapefile_path': '/content/aoi.shp',
    'buffer_km': 25.0,
    'gee_project_id': 'your-gcp-project-id',
    'models': ['EC-Earth3', 'CNRM-CM6-1', 'GFDL-ESM4'],
    'scenarios': ['ssp245', 'ssp585'],
    'baseline_start': '1990-01-01',
    'baseline_end': '2014-12-31',
    'hist_start': '1990-01-01',
    'future_intervals': [
        ('2026-01-01', '2050-12-31', 'Short', '2026-2050'),
        ('2051-01-01', '2075-12-31', 'Mid', '2051-2075'),
    ],
    'wet_months': [5, 6, 7, 8, 9, 10],
    'dry_months': [1, 2, 3, 4, 11, 12],
    'temp_thresholds': [30.0],
    'precip_thresholds': [20.0, 25.0],
    'return_periods': [100],
    'gev_n_bootstrap': 1000,
    'output_dir': '/content/drive/MyDrive/climate_output',
}
results = run_pipeline(params)
```

You'll need a Google Earth Engine account with a registered Cloud project
(`ee.Authenticate()` will prompt for this on first run).

## Key parameters

| Parameter | Meaning |
|---|---|
| `models` | CMIP6 GCM names, must match `model` values in `NASA/GDDP-CMIP6`. |
| `scenarios` | SSP scenario codes, e.g. `ssp245`, `ssp370`, `ssp585`. |
| `baseline_start` / `baseline_end` | Period used to train the QDM bias correction. |
| `future_intervals` | List of `(start, end, label, tag)` tuples defining future windows. |
| `wet_months` / `dry_months` | Month numbers (1–12) used for seasonal totals. |
| `temp_thresholds` / `precip_thresholds` | °C / mm thresholds for exceedance-day counts. |
| `return_periods` | Return periods (years) for GEV extreme-precipitation estimates. |
| `nquantiles`, `qdm_group`, `wet_thresh` | QDM tuning knobs (default 50 quantiles, grouped by month, 0.1 mm wet-day threshold). |

## Outputs

Written to `output_dir`, organized per variable rather than dumped in one folder:

```
output_dir/
├── climate_indices_summary.xlsx   # one row per index/period/domain: mean, p10, p90, headline stat
├── fanchart_tas.png                # historical + scenario spread over time
├── fanchart_pr.png
├── tas/
│   ├── ssp245/qdm_tas_<model>_ssp245_<tag>.nc      # per-model bias-corrected grids
│   ├── ssp585/qdm_tas_<model>_ssp585_<tag>.nc
│   └── ensemble/
│       ├── tas_<scenario>_<tag>_ensemble_mean.nc   # ensemble-mean grid across models
│       └── annual_mean_tas_<scenario>_<tag>_map.png
├── tasmax/  (same layout: ssp245/, ssp585/, ensemble/)
├── tasmin/  (same layout: ssp245/, ssp585/, ensemble/)
└── pr/
    ├── ssp245/  ssp585/            # per-model NetCDFs
    └── ensemble/
        ├── pr_<scenario>_<tag>_ensemble_mean.nc
        └── prcptot_<scenario>_<tag>_map.png
```

- **`{variable}/{scenario}/`** — the individual bias-corrected NetCDF for each GCM, so you
  can inspect or reuse a single model's output without touching the rest.
- **`{variable}/ensemble/`** — the model-ensemble mean NetCDF for each scenario/future
  period, plus the spatial-map PNG built from that same ensemble mean.
- Top-level `climate_indices_summary.xlsx` and the two fan charts stay at the root since
  they already summarize across variables/scenarios.

## Progress bars

Every long-running stage (reference fetch, QDM training, future projections, ensemble
means, index computation, spatial maps) shows a `tqdm` progress bar so you can see how
much work remains instead of staring at silent gaps between print statements.

## Notes on this version

This copy has three fixes applied on top of the original code, all verified against
current package versions:

1. **`gee_utils.py`** — removed a broken `from xclim.sdba.base import convert_calendar`
   import (that submodule no longer exists since `xclim` split its bias-adjustment code
   into the separate `xsdba` package). Calendar conversion now uses xarray's own
   `Dataset.convert_calendar()`.
2. **`plot_utils.py`** — added a missing `import rioxarray`, without which the `.rio`
   accessor used in `make_spatial_map()` doesn't exist and the spatial-maps step would
   crash.
3. **`qdm_utils.py`** — imports `xsdba` directly rather than through the deprecated
   `xclim.sdba` shim.

4. **`main.py`** — fixed a baseline-results bug: list-valued indices (e.g. per-month
   arrays, threshold-day counts) were left as bare lists instead of being wrapped in
   `{'mean','p10','p90'}` like every other index, which crashed the Excel-summary step
   with `AttributeError: 'list' object has no attribute 'get'` as soon as any temperature
   or precipitation threshold was configured. Fixed by wrapping consistently, while
   still leaving `gev_return_levels` (a genuine nested dict) untouched.

`requirements.txt` pins `xclim`, `xsdba`, and `xee` to compatible ranges and adds
`tqdm`; `run_interactive.ipynb` installs from `requirements.txt` (rather than a
hardcoded package list) so it can't drift out of sync again. Every long-running loop
now shows a `tqdm` progress bar, and output NetCDFs/plots are organized into
per-variable/scenario/ensemble folders instead of one flat directory (see "Outputs"
above).
