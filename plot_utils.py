import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from indices_utils import spatial_mean
from xclim import indices as xci

SCENARIO_COLORS = {'ssp245': '#e8a33d', 'ssp370': '#8a8a8a', 'ssp585': '#7a1f2b'}


# ──────────────────────────────────────────────────────────────────────────
# Fan charts (unchanged — these were already working)
# ──────────────────────────────────────────────────────────────────────────

def annual_series_from_grids(hist_cache, corrected_grids, var, scenario=None, tag=None, model_list=None):
    out = {}
    models = model_list or []
    for model in models:
        if scenario is None:
            da = hist_cache.get(var, {}).get(model)
        else:
            da = corrected_grids.get(var, {}).get(scenario, {}).get(tag, {}).get(model)
        if da is None:
            continue
        annual = (xci.tg_mean(spatial_mean(da), freq='YS') if var != 'pr'
                  else xci.precip_accumulation(spatial_mean(da), freq='YS'))
        out[model] = (annual['time'].dt.year.values, annual.values)
    return out


def plot_fan_chart(hist_cache, corrected_grids, var, ylabel, out_path, title,
                    scenarios, future_intervals, model_list):
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_series = annual_series_from_grids(hist_cache, corrected_grids, var, scenario=None, tag=None, model_list=model_list)
    if hist_series:
        years = sorted(set().union(*[set(y) for y, _ in hist_series.values()]))
        stacked = np.full((len(hist_series), len(years)), np.nan)
        for i, (y, v) in enumerate(hist_series.values()):
            idx = np.searchsorted(years, y)
            stacked[i, idx] = v
        ax.plot(years, np.nanmean(stacked, axis=0), color='black', label='Historical')
        ax.fill_between(years, np.nanpercentile(stacked, 10, axis=0),
                         np.nanpercentile(stacked, 90, axis=0), color='gray', alpha=0.25)

    for scenario in scenarios:
        all_years, all_mean, all_p10, all_p90 = [], [], [], []
        for start, end, label, tag in future_intervals:
            series = annual_series_from_grids(hist_cache, corrected_grids, var, scenario, tag, model_list)
            if not series:
                continue
            years = sorted(set().union(*[set(y) for y, _ in series.values()]))
            stacked = np.full((len(series), len(years)), np.nan)
            for i, (y, v) in enumerate(series.values()):
                idx = np.searchsorted(years, y)
                stacked[i, idx] = v
            all_years.extend(years)
            all_mean.extend(np.nanmean(stacked, axis=0))
            all_p10.extend(np.nanpercentile(stacked, 10, axis=0))
            all_p90.extend(np.nanpercentile(stacked, 90, axis=0))
        if all_years:
            color = SCENARIO_COLORS.get(scenario)
            ax.plot(all_years, all_mean, color=color, label=scenario.upper())
            ax.fill_between(all_years, all_p10, all_p90, color=color, alpha=0.2)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Spatial maps — rebuilt on the contourf + polygon-clip approach that was
# already proven to work in MappingfromNC.ipynb, instead of the rioxarray
# reprojection/clip path that was silently producing no output.
# ──────────────────────────────────────────────────────────────────────────

def _polygon_to_mpl_path(geom):
    """Convert a shapely Polygon/MultiPolygon into a matplotlib Path (incl. holes)."""
    all_verts, all_codes = [], []

    def process_ring(ring):
        xy = np.array(ring.coords)
        n = len(xy)
        all_verts.append(xy)
        all_codes.extend([Path.MOVETO] + [Path.LINETO] * (n - 2) + [Path.CLOSEPOLY])

    def process_polygon(poly):
        process_ring(poly.exterior)
        for interior in poly.interiors:
            process_ring(interior)

    if geom.geom_type == 'Polygon':
        process_polygon(geom)
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            process_polygon(poly)
    else:
        raise ValueError(f"Unsupported geometry type for clipping: {geom.geom_type}")

    return Path(np.vstack(all_verts), np.array(all_codes))


def _smooth_field(data, sigma=1.0):
    """Gap-aware gaussian smoothing — NaNs don't bleed into neighboring cells."""
    mask = np.isnan(data)
    filled = np.where(mask, 0.0, data)
    sm = gaussian_filter(filled, sigma=sigma)
    wt = gaussian_filter((~mask).astype(float), sigma=sigma)
    with np.errstate(invalid='ignore'):
        return np.where(wt > 0.01, sm / wt, np.nan)


def _add_background(ax, add_satellite, tile_zoom):
    """Satellite tiles if requested and reachable, else plain land/ocean fill.
    Never raises — a background failure should not stop the map from saving.
    """
    if add_satellite:
        try:
            tiler = cimgt.GoogleTiles(
                url="https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                desired_tile_form="RGB")
            ax.add_image(tiler, tile_zoom)
            return
        except Exception as e:
            print(f"⚠️ Satellite tiles unavailable ({e}); falling back to plain background")

    ax.set_facecolor('#dbe7f0')
    ax.add_feature(cfeature.LAND, facecolor='#f2f0e8', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#dbe7f0', zorder=0)


def make_spatial_map(index_da_2d, geom_native, out_path, title, cmap='viridis',
                      smooth_sigma=1.0, add_satellite=False, tile_zoom=8):
    """
    Render one clipped, smoothed spatial map from an in-memory (lat, lon)
    DataArray and save it to out_path.

    index_da_2d : xr.DataArray with dims ('lat', 'lon')
    geom_native : shapely Polygon/MultiPolygon (unbuffered AOI, EPSG:4326)
    """
    da = index_da_2d.sortby('lat').sortby('lon')
    lon = da['lon'].values
    lat = da['lat'].values
    vals_smooth = _smooth_field(da.values, sigma=smooth_sigma)

    shp = gpd.GeoSeries([geom_native], crs='EPSG:4326')
    minx, miny, maxx, maxy = shp.total_bounds
    buf = 0.25
    extent = [minx - buf, maxx + buf, miny - buf, maxy + buf]

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7, 6), facecolor='white')
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=proj)

    _add_background(ax, add_satellite, tile_zoom)

    cf = ax.contourf(lon, lat, vals_smooth, levels=60, cmap=cmap,
                      transform=proj, extend='both', zorder=3)

    # Clip strictly to the AOI polygon (this is the step that was failing
    # silently under rioxarray — matplotlib-path clipping is what actually
    # worked in the notebook).
    clip_path = _polygon_to_mpl_path(geom_native)
    patch = PathPatch(clip_path, transform=ax.transData)
    artists = getattr(cf, 'collections', None) or [
        a for a in cf.get_children() if hasattr(a, 'set_clip_path')
    ]
    for artist in artists:
        artist.set_clip_path(patch)

    shp.boundary.plot(ax=ax, color='black', linewidth=1.2, transform=proj, zorder=6)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='grey', alpha=0.5,
                       linestyle='--', crs=proj)
    gl.top_labels = False
    gl.right_labels = False

    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.03)
    cb.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"🗺️  Map saved: {out_path}")
