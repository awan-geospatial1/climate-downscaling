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

def _smooth_and_clip(index_da_2d, geom_native, upsample_factor=4, smooth_sigma=1.0):
    """Upsample a (lat,lon) grid, Gaussian-smooth it, then hard-clip to geom_native
    (cells outside the boundary become NaN and are dropped from the bounding box).
    Shared by make_spatial_map() and make_index_panel() so both produce identical
    smoothing/masking behaviour."""
    da = index_da_2d.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False)
    da = da.rio.write_crs('EPSG:4326')
    new_lat = np.linspace(float(da.lat.min()), float(da.lat.max()), da.sizes['lat']*upsample_factor)
    new_lon = np.linspace(float(da.lon.min()), float(da.lon.max()), da.sizes['lon']*upsample_factor)
    da_fine = da.interp(lat=new_lat, lon=new_lon, method='linear')
    smoothed = gaussian_filter(np.nan_to_num(da_fine.values, nan=np.nanmean(da_fine.values)), sigma=smooth_sigma)
    da_smooth = xr.DataArray(smoothed, coords={'lat':new_lat,'lon':new_lon}, dims=['lat','lon'])
    da_smooth = da_smooth.rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.write_crs('EPSG:4326')
    da_clipped = da_smooth.rio.clip([mapping(geom_native)], crs='EPSG:4326', drop=True, all_touched=True)
    return da_clipped

def make_spatial_map(index_da_2d, geom_native, out_path, title, cmap='viridis',
                     smooth_sigma=1.0, upsample_factor=4):
    da_clipped = _smooth_and_clip(index_da_2d, geom_native, upsample_factor, smooth_sigma)
    fig, ax = plt.subplots(figsize=(7,6))
    da_clipped.plot(ax=ax, cmap=cmap, add_colorbar=True)
    gpd.GeoSeries([geom_native], crs='EPSG:4326').boundary.plot(ax=ax, color='black', linewidth=1)
    ax.set_title(title); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def make_index_panel(ref_da, panel_grids, geom_native, out_path, title, subtitle,
                     cbar_label, cmap='Blues', smooth_sigma=1.0, upsample_factor=4):
    """
    Build one 'AHP-style' panel figure for a single index: a centered Reference
    (baseline) tile on top, then one row per scenario x one column per future
    period below -- all smoothed, clipped to geom_native, and sharing one
    colorbar, with a Mean/P99 stat box on every tile.

    ref_da: 2D (lat,lon) ensemble grid for the reference/baseline period.
    panel_grids: {scenario: {tag: 2D DataArray}} of ensemble-mean grids for
                 each scenario x future-period combination.
    """
    scenarios = list(panel_grids.keys())
    tags = sorted({tag for s in panel_grids.values() for tag in s.keys()})
    if not tags:
        tags = ['']
    n_rows = 1 + len(scenarios)
    n_cols = max(len(tags), 1)

    clipped_ref = _smooth_and_clip(ref_da, geom_native, upsample_factor, smooth_sigma)
    clipped = {}
    all_vals = [clipped_ref.values]
    for s in scenarios:
        clipped[s] = {}
        for tag in tags:
            da = panel_grids.get(s, {}).get(tag)
            if da is None:
                continue
            c = _smooth_and_clip(da, geom_native, upsample_factor, smooth_sigma)
            clipped[s][tag] = c
            all_vals.append(c.values)
    vmin = float(np.nanmin([np.nanmin(v) for v in all_vals]))
    vmax = float(np.nanmax([np.nanmax(v) for v in all_vals]))

    fig = plt.figure(figsize=(4.2 * n_cols, 3.6 * n_rows + 1.2))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.15)
    im = None

    def _draw(ax, da, panel_title):
        nonlocal im
        im = da.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
        gpd.GeoSeries([geom_native], crs='EPSG:4326').boundary.plot(ax=ax, color='black', linewidth=0.8)
        ax.set_title(panel_title, fontsize=10, fontweight='bold')
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticks([]); ax.set_yticks([])
        mean_v = float(np.nanmean(da.values))
        p99_v = float(np.nanpercentile(da.values, 99))
        ax.text(0.98, 0.03, f"Mean : {mean_v:.1f}\nP99 : {p99_v:.1f}", transform=ax.transAxes,
                fontsize=7, color='white', ha='right', va='bottom',
                bbox=dict(facecolor='black', alpha=0.55, boxstyle='round,pad=0.25'))

    ref_col = (n_cols - 1) // 2
    _draw(fig.add_subplot(gs[0, ref_col]), clipped_ref, 'Reference (baseline)')
    for c in range(n_cols):
        if c != ref_col:
            fig.add_subplot(gs[0, c]).axis('off')

    for r, s in enumerate(scenarios, start=1):
        for c, tag in enumerate(tags):
            ax = fig.add_subplot(gs[r, c])
            da = clipped.get(s, {}).get(tag)
            if da is None:
                ax.axis('off')
                continue
            _draw(ax, da, tag)
        fig.text(0.02, 1 - (r + 0.5) / n_rows, s.upper(), rotation=90,
                 va='center', ha='center', fontsize=10, fontweight='bold')

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.6])
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.955, subtitle, ha='center', fontsize=9, color='#555555')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
