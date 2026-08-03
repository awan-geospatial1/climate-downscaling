"""
ensemble_utils.py - Multi-model ensemble aggregation across the daily
bias-corrected grids the pipeline already holds in memory
(`corrected_grids[var][scenario][tag]`, a dict of model -> xr.DataArray
with dims (time, lat, lon)).

Added because the pipeline was computing an ensemble mean only transiently,
in memory, for two hardcoded spatial-map indices (`annual_mean_tas`,
`prcptot`) inside main.py's spatial-maps loop -- never for every variable,
never for every scenario/period, and never saved to disk. Anything that
needed a general per-variable ensemble grid (e.g. other analysis scripts
reading a `<scenario>/<tag>/ensemble/ensemble_mean_<var>.nc` file) had
nothing to read, even though the README already documented this file as
part of the output layout.

Two aggregations:
  - compute_ensemble_mean : elementwise mean across models, every day,
    every cell. Used for ALL FOUR variables (tas, tasmax, tasmin, pr).
  - compute_ensemble_max  : elementwise MAXIMUM across models, every day,
    every cell -- i.e. at each grid cell and day, the single most extreme
    value any model in the ensemble produced. Used for PRECIPITATION ONLY,
    since an "ensemble max" grid is specifically useful for extreme-value /
    hazard-style analysis (Max_Precip / return-period style reporting),
    where you want the envelope of the most extreme model rather than the
    smoothed-out mean. It is not meaningful the same way for temperature
    (there's no equivalent "worst case" framing requested for tas/tasmax/
    tasmin), so this is intentionally precip-only per the request.

Both preserve the full (time, lat, lon) daily resolution, not just an
annual/period summary — that keeps the saved NetCDF equally useful for
anything downstream that wants daily data (e.g. daily_spatial_ensemble-
style series, or annual aggregation done later at read time), rather than
baking in one specific aggregation window up front.
"""
import numpy as np
import pandas as pd
import xarray as xr


def compute_ensemble_mean(grids_by_model):
    """dict[model] -> xr.DataArray (time, lat, lon), all sharing the same
    grid (they're all already regridded to the reference grid upstream in
    the pipeline) -> single xr.DataArray: elementwise mean across models.
    """
    if not grids_by_model:
        raise ValueError("compute_ensemble_mean: no models supplied.")
    stacked = xr.concat(list(grids_by_model.values()), dim='model')
    return stacked.mean(dim='model', skipna=True)


def compute_ensemble_max(grids_by_model):
    """Same shape of input as compute_ensemble_mean, but takes the
    elementwise MAXIMUM across models instead of the mean -- the most
    extreme value any single model produced, at every cell and day.
    """
    if not grids_by_model:
        raise ValueError("compute_ensemble_max: no models supplied.")
    stacked = xr.concat(list(grids_by_model.values()), dim='model')
    return stacked.max(dim='model', skipna=True)


def save_ensemble_netcdf(da, out_path, units=None, extra_attrs=None):
    """Save an ensemble grid to NetCDF, matching the same
    `.load().to_netcdf(...)` pattern already used for per-model grids
    elsewhere in the pipeline, so file sizes/behavior are consistent.
    """
    da = da.copy()
    if units is not None:
        da.attrs['units'] = units
    if extra_attrs:
        da.attrs.update(extra_attrs)
    da.load().to_netcdf(out_path)
    return out_path


def ensemble_spatial_series(da, units=None):
    """1-D pandas Series of the spatial mean of an ensemble (time, lat, lon)
    grid, indexed by date -- the ensemble equivalent of
    indices_utils.daily_spatial_series, but operating on an already-reduced
    ensemble DataArray instead of a dict of per-model grids (since the
    ensemble reduction already happened in compute_ensemble_mean/max).
    Converts Kelvin to Celsius for readability, same convention as the rest
    of the pipeline's daily-series helpers.
    """
    dims = [d for d in ('lat', 'lon') if d in da.dims]
    s = (da.mean(dim=dims, skipna=True) if dims else da).to_series()
    if units == 'K':
        s = s - 273.15
    s.index = pd.to_datetime(s.index).normalize()
    s.index.name = 'date'
    return s
