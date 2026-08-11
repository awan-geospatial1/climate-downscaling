# Climate Downscaling — Quantile Delta Mapping on Google Earth Engine

Bias-correct and statistically downscale CMIP6 climate projections against a
high-resolution observational reference (CHIRPS for precipitation, ERA5-Land
for temperature), for any area of interest defined by a shapefile — all
computed server-side on Google Earth Engine and post-processed locally with
`xarray`/`xclim`.

The pipeline trains a **Quantile Delta Mapping (QDM)** transfer function per
model and variable, applies it to both the historical record and future
scenario periods, derives standard climate indices, and exports tabular
summaries, ensemble fan charts, and spatial maps — ready for a climate risk
assessment or downscaling report.

## What it does

1. **Load the area of interest** from a shapefile, buffer it, and build a
   Google Earth Engine geometry (`main.load_shapefile`).
2. **Fetch reference and CMIP6 data** for each variable over Earth Engine,
   clipped to the AOI (`gee_utils.fetch_reference`, `gee_utils.fetch_cmip6`),
   then regrid the model data onto the reference grid
   (`gee_utils.regrid_to_reference`).
3. **Train QDM** per variable/model on the baseline period and **apply it**
   to the historical and future periods (`qdm_utils.train_qdm`,
   `qdm_utils.apply_qdm`), with an optional wet-day frequency adjustment for
   precipitation (`qdm_utils.adjust_wet_day_frequency`).
4. **Compute climate indices** — temperature and precipitation — over the
   baseline and every scenario/interval combination, then aggregate across
   the model ensemble (`indices_utils.compute_temperature_indices`,
   `compute_precipitation_indices`, `aggregate_across_models`). This
   includes extreme value (GEV) return-level estimation with bootstrap
   confidence bounds.
5. **Export results**:
   - Bias-corrected NetCDF grids per variable/model/scenario/interval
   - An Excel summary (`climate_indices_summary.xlsx`) of every index with
     mean/p10/p90 and a configurable "headline" statistic
   - Ensemble fan charts for temperature and precipitation
     (`plot_utils.plot_fan_chart`)
   - Spatial maps of ensemble-mean indices (`plot_utils.make_spatial_map`)

## Variables and reference datasets

| Variable | Description       | Reference collection            | Reference band          |
|----------|--------------------|----------------------------------|--------------------------|
| `tas`    | Mean temperature   | `ECMWF/ERA5_LAND/DAILY_AGGR`     | `temperature_2m`         |
| `tasmax` | Max temperature    | `ECMWF/ERA5_LAND/DAILY_AGGR`     | `temperature_2m_max`     |
| `tasmin` | Min temperature    | `ECMWF/ERA5_LAND/DAILY_AGGR`     | `temperature_2m_min`     |
| `pr`     | Precipitation      | `UCSB-CHG/CHIRPS/DAILY`          | `precipitation`          |

CMIP6 model data is pulled from Earth Engine's CMIP6 archive for whichever
models and scenarios you specify. Per-variable settings (units, QDM kind —
additive for temperature, multiplicative for precipitation — clipping
bounds, fill values, and native scale) live in `config.py` and can be
overridden.

## Repository layout

| File                   | Purpose                                                              |
|-------------------------|-----------------------------------------------------------------------|
| `main.py`               | Orchestrates the full pipeline (`run_pipeline`)                      |
| `config.py`              | Per-variable settings and default QDM parameters                     |
| `gee_utils.py`           | Earth Engine data fetching, regridding, and cleanup helpers          |
| `qdm_utils.py`           | QDM training, application, and wet-day frequency adjustment          |
| `indices_utils.py`       | Temperature/precipitation index calculation and ensemble aggregation |
| `plot_utils.py`          | Fan chart and spatial map plotting                                   |
| `run_interactive.ipynb`  | Interactive notebook front-end for running the pipeline              |
| `requirements.txt`       | Python dependencies                                                  |

## Requirements

- A Google Earth Engine account with a Cloud project enabled for Earth
  Engine access
- Python 3.9+ (packages: `earthengine-api`, `xarray`, `xclim`, `xee`,
  `netCDF4`, `dask`, `geopandas`, `rioxarray`, `shapely`, `pyproj`, `scipy`,
  `matplotlib`, `openpyxl`, `pandas`, `numpy`, `ipywidgets`, `xsdba`)

Install everything with:

```bash
pip install -r requirements.txt
```

## Setup

1. **Authenticate with Earth Engine** (one-time, browser-based):

   ```python
   import ee
   ee.Authenticate()
   ```

   `main.run_pipeline` will call `ee.Initialize()` automatically and fall
   back to `ee.Authenticate()` + `ee.Initialize(project=...)` if needed.

2. **Prepare a shapefile** of your area of interest (any CRS — it's
   reprojected to EPSG:4326 automatically).

## Usage

Run the pipeline programmatically:

```python
from main import run_pipeline

params = {
    "shapefile_path": "aoi/my_area.shp",
    "buffer_km": 25.0,                       # AOI buffer, default 25 km
    "gee_project_id": "my-gee-project",
    "models": ["ACCESS-CM2", "MIROC6"],      # CMIP6 model IDs
    "scenarios": ["ssp245", "ssp585"],
    "baseline_start": "1995-01-01",
    "baseline_end": "2014-12-31",
    "hist_start": "1985-01-01",              # optional, defaults to baseline_start
    "future_intervals": [
        ("2041-01-01", "2060-12-31", "Mid-century", "2041-2060"),
        ("2081-01-01", "2100-12-31", "End-century", "2081-2100"),
    ],
    "wet_months": [6, 7, 8, 9],
    "dry_months": [12, 1, 2],
    "temp_thresholds": {...},
    "precip_thresholds": {...},
    "return_periods": [10, 25, 50, 100],
    "gev_n_bootstrap": 1000,
    "output_dir": "outputs/",
}

results = run_pipeline(params)
```

Or use the interactive notebook, `run_interactive.ipynb`, which wraps the
same pipeline with widgets for picking the shapefile, models, scenarios,
and date ranges.

### Key parameters

| Key                 | Description                                                        |
|----------------------|----------------------------------------------------------------------|
| `shapefile_path`     | Path to the AOI shapefile                                          |
| `buffer_km`          | Buffer distance applied to the AOI before clipping                 |
| `gee_project_id`     | Google Cloud project registered for Earth Engine                   |
| `models`             | List of CMIP6 model IDs to downscale                                |
| `scenarios`          | List of SSP scenarios (e.g. `ssp245`, `ssp585`)                     |
| `baseline_start/end` | Historical calibration period used to train QDM                     |
| `future_intervals`   | List of `(start, end, label, tag)` tuples for future periods         |
| `nquantiles`         | Number of quantiles for QDM (default 50)                            |
| `qdm_group`          | Grouping for QDM training, e.g. `time.month` (default)              |
| `wet_thresh`         | Wet-day threshold (mm) for precipitation frequency adjustment        |
| `return_periods`     | Return periods (years) for GEV extreme value analysis               |
| `output_dir`         | Directory for all exported NetCDF, Excel, and plot outputs           |

## Outputs

Running the pipeline populates `output_dir` with:

- `qdm_<var>_<model>_<scenario>_<tag>.nc` — bias-corrected NetCDF grids
- `climate_indices_summary.xlsx` — all temperature/precipitation indices
  across baseline and future periods, with mean/p10/p90 and headline values
- `fanchart_tas.png`, `fanchart_pr.png` — ensemble fan charts
- `spatial_maps/*.png` — ensemble-mean spatial maps per index/scenario/period

## Method notes

- **QDM (Quantile Delta Mapping)** preserves the projected change signal
  from each climate model while correcting its distributional bias against
  the observational reference, applied additively for temperature and
  multiplicatively for precipitation.
- **Wet-day frequency adjustment** corrects the tendency of climate models
  to simulate too many low-intensity "drizzle" days relative to observations.
- **GEV return-level estimation** uses a bootstrap to quantify uncertainty
  in extreme precipitation return periods.
