import os
import numpy as np
import pandas as pd
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
_FALLBACK_PALETTE = ['#3b6ea5', '#4c9a5b', '#a34ca0', '#c2703d', '#3d9aa3']

def _scenario_color(scenario, scenarios_in_run):
    """Color for a scenario. Falls back to a consistent palette slot for any
    scenario not in SCENARIO_COLORS (e.g. if only ssp245/ssp585 are used and
    not ssp370, or if a scenario name outside the usual three is passed)."""
    if scenario in SCENARIO_COLORS:
        return SCENARIO_COLORS[scenario]
    unknown = [s for s in scenarios_in_run if s not in SCENARIO_COLORS]
    return _FALLBACK_PALETTE[unknown.index(scenario) % len(_FALLBACK_PALETTE)]


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
    lines_plotted = 0

    try:
        hist_series = annual_series_from_grids(hist_cache, corrected_grids, var, scenario=None, tag=None, model_list=model_list)
    except Exception as e:
        print(f"⚠️ fan chart ({var}): historical series failed: {e}")
        hist_series = {}

    if hist_series:
        years = sorted(set().union(*[set(y) for y, _ in hist_series.values()]))
        stacked = np.full((len(hist_series), len(years)), np.nan)
        for i, (y, v) in enumerate(hist_series.values()):
            idx = np.searchsorted(years, y)
            stacked[i, idx] = v
        hist_mean = np.nanmean(stacked, axis=0)
        if np.isfinite(hist_mean).any():
            ax.plot(years, hist_mean, color='black', label='Historical')
            ax.fill_between(years, np.nanpercentile(stacked, 10, axis=0),
                             np.nanpercentile(stacked, 90, axis=0), color='gray', alpha=0.25)
            lines_plotted += 1
        else:
            print(f"⚠️ fan chart ({var}): historical series is all-NaN, skipping that line")

    for scenario in scenarios:
        try:
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
            if all_years and np.isfinite(all_mean).any():
                color = _scenario_color(scenario, scenarios)
                ax.plot(all_years, all_mean, color=color, label=scenario.upper())
                ax.fill_between(all_years, all_p10, all_p90, color=color, alpha=0.2)
                lines_plotted += 1
            elif not all_years:
                print(f"⚠️ fan chart ({var}): no corrected data for scenario '{scenario}' — nothing to plot for it")
            else:
                print(f"⚠️ fan chart ({var}): scenario '{scenario}' series is all-NaN, skipping that line")
        except Exception as e:
            print(f"⚠️ fan chart ({var}): scenario '{scenario}' failed: {e}")

    if lines_plotted == 0:
        # Guarantee a file is still produced, with an explicit message,
        # rather than silently saving a blank/empty-looking chart.
        ax.text(0.5, 0.5, 'No plottable data for this variable\n(see console warnings above)',
                ha='center', va='center', transform=ax.transAxes, fontsize=11, color='#a33')
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.legend()

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"📈 Fan chart saved ({lines_plotted} line(s)): {out_path}")


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


# ──────────────────────────────────────────────────────────────────────────
# Scenario-comparison bar charts (new)
# ──────────────────────────────────────────────────────────────────────────

def plot_index_comparison(df, index_name, out_path, ylabel=None):
    """
    Grouped bar chart comparing one index's headline value across every
    scenario, for every future period, with p10-p90 as an error bar.
    `df` is the same long-format summary DataFrame written to Excel
    (columns: period, domain, index, mean, p10, p90, headline_value).
    Skips (returns False) if the index isn't present for >=2 periods.
    """
    sub = df[df['index'] == index_name].copy()
    if sub.empty:
        return False

    # period looks like 'Baseline' or '<scenario>_<tag>'
    def split_period(p):
        if p == 'Baseline':
            return 'Baseline', 'Baseline'
        parts = p.split('_', 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (p, '')

    sub[['scenario', 'tag']] = sub['period'].apply(lambda p: pd.Series(split_period(p)))
    tags = list(dict.fromkeys(sub['tag']))  # preserve first-seen order
    scenarios = [s for s in dict.fromkeys(sub['scenario']) if s != 'Baseline']

    fig, ax = plt.subplots(figsize=(9, 5.5))
    baseline_row = sub[sub['scenario'] == 'Baseline']
    if not baseline_row.empty:
        ax.axhline(baseline_row['mean'].iloc[0], color='black', linestyle='--',
                   linewidth=1, label='Baseline')

    width = 0.8 / max(len(scenarios), 1)
    x = np.arange(len(tags))
    for i, scenario in enumerate(scenarios):
        vals, lo, hi = [], [], []
        for tag in tags:
            row = sub[(sub['scenario'] == scenario) & (sub['tag'] == tag)]
            if row.empty:
                vals.append(np.nan); lo.append(0); hi.append(0)
            else:
                vals.append(row['mean'].iloc[0])
                lo.append(row['mean'].iloc[0] - row['p10'].iloc[0])
                hi.append(row['p90'].iloc[0] - row['mean'].iloc[0])
        offset = (i - (len(scenarios) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=scenario.upper(),
               color=_scenario_color(scenario, scenarios), yerr=[lo, hi], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel(ylabel or index_name)
    ax.set_title(f'{index_name} — scenario comparison')
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


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

    # Clip strictly to the AOI polygon.
    # NOTE: on matplotlib >=3.8, QuadContourSet no longer exposes `.collections`
    # (accessing it raises or is deprecated), AND `cf.get_children()` returns
    # an EMPTY list because the filled contours live on the ContourSet object
    # itself now, not on child artists. The old fallback
    # `[a for a in cf.get_children() if hasattr(a, 'set_clip_path')]` therefore
    # silently produced an empty artist list on modern matplotlib — no error,
    # no clip ever applied, every map rendered as an unclipped rectangle.
    # Fix: try the pre-3.8 `.collections` path first, and always also try
    # clipping the ContourSet object directly (confirmed to work on 3.10+).
    clip_path = _polygon_to_mpl_path(geom_native)
    patch = PathPatch(clip_path, transform=ax.transData)
    artists = list(getattr(cf, 'collections', None) or [])
    if hasattr(cf, 'set_clip_path'):
        artists.append(cf)
    for artist in artists:
        try:
            artist.set_clip_path(patch)
        except Exception:
            pass

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
