import matplotlib.pyplot as plt
import xarray as xr, numpy as np, geopandas as gpd, os
import rioxarray  # noqa: F401  -- required to register the .rio accessor on DataArray/Dataset
from scipy.ndimage import gaussian_filter
from shapely.geometry import mapping
from indices_utils import spatial_mean
from xclim import indices as xci

SCENARIO_COLORS = {'ssp245':'#e8a33d', 'ssp370':'#8a8a8a', 'ssp585':'#7a1f2b'}

def annual_series_from_grids(hist_cache, corrected_grids, var, scenario=None, tag=None, model_list=None):
    out = {}
    models = model_list or []
    for model in models:
        if scenario is None:
            da = hist_cache.get(var, {}).get(model)
        else:
            da = corrected_grids.get(var, {}).get(scenario, {}).get(tag, {}).get(model)
        if da is None: continue
        annual = (xci.tg_mean(spatial_mean(da), freq='YS') if var != 'pr'
                  else xci.precip_accumulation(spatial_mean(da), freq='YS'))
        out[model] = (annual['time'].dt.year.values, annual.values)
    return out

def plot_fan_chart(hist_cache, corrected_grids, var, ylabel, out_path, title,
                   scenarios, future_intervals, model_list):
    fig, ax = plt.subplots(figsize=(10,6))
    hist_series = annual_series_from_grids(hist_cache, corrected_grids, var, scenario=None, tag=None, model_list=model_list)
    if hist_series:
        years = sorted(set().union(*[set(y) for y,_ in hist_series.values()]))
        stacked = np.full((len(hist_series), len(years)), np.nan)
        for i,(y,v) in enumerate(hist_series.values()):
            idx = np.searchsorted(years, y); stacked[i, idx] = v
        ax.plot(years, np.nanmean(stacked, axis=0), color='black', label='Historical')
        ax.fill_between(years, np.nanpercentile(stacked,10,axis=0),
                        np.nanpercentile(stacked,90,axis=0), color='gray', alpha=0.25)
    for scenario in scenarios:
        all_years, all_mean, all_p10, all_p90 = [], [], [], []
        for start,end,label,tag in future_intervals:
            series = annual_series_from_grids(hist_cache, corrected_grids, var, scenario, tag, model_list)
            if not series: continue
            years = sorted(set().union(*[set(y) for y,_ in series.values()]))
            stacked = np.full((len(series), len(years)), np.nan)
            for i,(y,v) in enumerate(series.values()):
                idx = np.searchsorted(years, y); stacked[i, idx] = v
            all_years.extend(years)
            all_mean.extend(np.nanmean(stacked, axis=0))
            all_p10.extend(np.nanpercentile(stacked,10,axis=0))
            all_p90.extend(np.nanpercentile(stacked,90,axis=0))
        if all_years:
            color = SCENARIO_COLORS.get(scenario)
            ax.plot(all_years, all_mean, color=color, label=scenario.upper())
            ax.fill_between(all_years, all_p10, all_p90, color=color, alpha=0.2)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)

def make_spatial_map(index_da_2d, geom_native, out_path, title, cmap='viridis',
                     smooth_sigma=1.0, upsample_factor=4):
    da = index_da_2d.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False)
    da = da.rio.write_crs('EPSG:4326')
    new_lat = np.linspace(float(da.lat.min()), float(da.lat.max()), da.sizes['lat']*upsample_factor)
    new_lon = np.linspace(float(da.lon.min()), float(da.lon.max()), da.sizes['lon']*upsample_factor)
    da_fine = da.interp(lat=new_lat, lon=new_lon, method='linear')
    smoothed = gaussian_filter(np.nan_to_num(da_fine.values, nan=np.nanmean(da_fine.values)), sigma=smooth_sigma)
    da_smooth = xr.DataArray(smoothed, coords={'lat':new_lat,'lon':new_lon}, dims=['lat','lon'])
    da_smooth = da_smooth.rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.write_crs('EPSG:4326')
    da_clipped = da_smooth.rio.clip([mapping(geom_native)], crs='EPSG:4326', drop=True, all_touched=True)
    fig, ax = plt.subplots(figsize=(7,6))
    da_clipped.plot(ax=ax, cmap=cmap, add_colorbar=True)
    gpd.GeoSeries([geom_native], crs='EPSG:4326').boundary.plot(ax=ax, color='black', linewidth=1)
    ax.set_title(title); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
