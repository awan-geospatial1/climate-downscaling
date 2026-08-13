import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def _nice_ticks(vmin, vmax, n=9):
    """
    n evenly-spaced tick values spanning exactly [vmin, vmax], rounded to a
    sensible number of significant figures instead of matplotlib's default
    colorbar tick locator, which can land on ugly, hard-to-read decimals
    (e.g. "5.136, 5.184, 5.232..."). Always includes the exact vmin/vmax
    endpoints (rounding only the interior ticks) so the legend's stated
    range still matches what the color scale actually spans.
    """
    raw = np.linspace(vmin, vmax, n)
    span = vmax - vmin
    if span <= 0:
        return raw
    # Pick a rounding precision from the span's order of magnitude so ticks
    # read as clean numbers (e.g. spans of 100s round to whole numbers,
    # spans under 1 round to 2-3 decimals) without collapsing distinct
    # ticks into duplicates for very small spans.
    decimals = max(0, int(np.ceil(-np.log10(span / n))) + 1)
    ticks = np.round(raw, decimals)
    ticks[0], ticks[-1] = vmin, vmax
    return ticks


def _hatch_masked_region(ax, lon, lat, mask, proj, hatch='///', zorder=4):
    """
    Overlay diagonal hatching over cells where `mask` is True (e.g. pct-change
    is undefined because baseline is ~0) — previously these NaN cells were
    just left uncolored (blank, matching the basemap), which is easy to
    mistake for "no interesting data" rather than "this value can't be
    expressed on this legend, at all" (division by ~0 baseline). Hatching
    makes that distinction visible without inventing a fake color/value for
    a mathematically undefined quantity — this is what "all the values
    inside the masked region" should look like: masked cells are visibly
    marked as masked rather than silently dropped.
    """
    if not mask.any():
        return
    try:
        ax.contourf(lon, lat, mask.astype(float), levels=[0.5, 1.5], colors='none',
                    hatches=[hatch], transform=proj, zorder=zorder)
    except Exception:
        pass


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
    # FIX: sortby() does not guarantee axis ORDER, only sort direction along
    # each named axis — the reducers upstream (e.g. xci.tg_mean(da).mean
    # (dim='time')) can hand back a DataArray whose dims are ('lon', 'lat')
    # instead of ('lat', 'lon'). .values then comes out transposed relative
    # to what contourf(lon, lat, z) requires (z must be (len(lat), len(lon))),
    # which is exactly what caused every single spatial/agreement map in a
    # real run to fail with "Length of x (N) must match number of columns
    # in z (M)". An explicit .transpose() makes the axis order unconditional
    # regardless of what the reducer returned.
    da = index_da_2d.sortby('lat').sortby('lon').transpose('lat', 'lon')
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


def _clip_to_aoi(cf, geom_native, ax):
    """Shared clip logic — see the matplotlib >=3.8 note in make_spatial_map."""
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


def make_composite_grid_map(baseline_da, scenario_grids, scenario_order, period_order,
                             geom_native, districts_gdf, out_path,
                             var_title, unit_baseline, unit_delta,
                             cmap_baseline='YlOrRd', diverging_cmap='RdBu_r',
                             pct_change=False, smooth_sigma=1.0,
                             add_satellite=True, tile_zoom=8):
    """
    Report-style composite: one large baseline panel (own colorbar) plus
    a grid of scenario x period change panels sharing one diverging colorbar.

    Rows = scenario_order, columns = period_order — both are just lists, so
    this auto-sizes to however many periods were selected (3, 4, 5, ...)
    without any other change.

    baseline_da    : xr.DataArray (lat, lon) — ensemble-mean baseline field.
    scenario_grids : {scenario: {tag: xr.DataArray(lat, lon)}} — ensemble-mean
                      future field for every scenario/period cell. A missing
                      cell is simply left blank rather than erroring.
    scenario_order : [scenario, ...] top-to-bottom row order.
    period_order   : [(tag, label), ...] left-to-right column order; `label`
                      is what's printed as the column header (e.g. '2021-2040').
    districts_gdf  : original, non-dissolved AOI GeoDataFrame — draws each
                      district/sub-basin's own boundary in white, like the
                      reference figure. Pass None to only draw the outer edge.
    pct_change     : True -> cells show % change vs baseline (precip-style);
                      False -> cells show absolute change (temperature-style).
    """
    n_rows, n_cols = len(scenario_order), len(period_order)
    proj = ccrs.PlateCarree()

    shp_outer = gpd.GeoSeries([geom_native], crs='EPSG:4326')
    minx, miny, maxx, maxy = shp_outer.total_bounds
    buf = 0.25
    extent = [minx - buf, maxx + buf, miny - buf, maxy + buf]

    # FIX: force ('lat','lon') axis order explicitly — see make_spatial_map's
    # comment above for why sortby() alone isn't enough.
    base_sorted = baseline_da.sortby('lat').sortby('lon').transpose('lat', 'lon')
    base_smooth = _smooth_field(base_sorted.values, sigma=smooth_sigma)
    lonb, latb = base_sorted['lon'].values, base_sorted['lat'].values

    # First pass: compute every cell's delta field and pool them so the whole
    # grid can share one color range that represents every real value in it.
    cell_fields, all_finite_vals = {}, []
    for scenario in scenario_order:
        for tag, _label in period_order:
            da = scenario_grids.get(scenario, {}).get(tag)
            if da is None:
                continue
            da_sorted = da.sortby('lat').sortby('lon').transpose('lat', 'lon')
            vals_smooth = _smooth_field(da_sorted.values, sigma=smooth_sigma)
            if pct_change:
                with np.errstate(divide='ignore', invalid='ignore'):
                    delta = np.where(np.abs(base_smooth) > 1e-6,
                                      (vals_smooth - base_smooth) / np.abs(base_smooth) * 100,
                                      np.nan)
            else:
                delta = vals_smooth - base_smooth
            cell_fields[(scenario, tag)] = (da_sorted['lon'].values, da_sorted['lat'].values, delta)
            finite = delta[np.isfinite(delta)]
            if finite.size:
                all_finite_vals.append(finite)

    # FIX: this used to clip vmax to the 98th percentile of |delta|, which
    # meant the top ~2% of cells (by magnitude) got flattened into the
    # colorbar's "extend" cap color instead of showing their real value —
    # exactly what produced the solid, undifferentiated dark-navy blocks in
    # earlier renders (several districts with genuinely different large
    # negative % changes were all indistinguishable off-scale-cap color).
    # Using the true max means every real value in the data gets its own
    # distinct color on the legend; nothing is clipped, so 'extend' is no
    # longer needed either.
    if all_finite_vals:
        vmax = float(np.nanmax(np.abs(np.concatenate(all_finite_vals)))) or 1.0
    else:
        vmax = 1.0
    vmin = -vmax

    fig_h = 2.9 * n_rows + 1.4
    title_band_in = 0.9  # fixed absolute height regardless of n_rows, so a 1-row
                          # grid doesn't compress/overlap the two title lines
    fig_h_total = fig_h + title_band_in
    # FIX: the baseline panel spans gs[:, 0] -- ALL n_rows -- but this used
    # to hard-code its width_ratio to a flat 1.5 regardless of n_rows, and
    # figure width only grew with n_cols. Every regular column is 1 row tall
    # with width_ratio=1; the baseline column is n_rows tall, so keeping its
    # width_ratio fixed at 1.5 meant its cell got proportionally TALLER and
    # NARROWER as more scenario rows were added. cartopy's aspect-locked
    # GeoAxes can't stretch to fill a mismatched cell — it shrinks to the
    # correct lat/lon aspect and centers itself, which is exactly what left
    # growing blank margins above and below the baseline map (worse the
    # more scenarios you add). Scaling width_ratio by n_rows keeps the same
    # per-row aspect as the original 1-row design; the figure width has to
    # grow with it too (in the same 2.9in/unit the row heights already use)
    # so this doesn't just steal width from the regular columns instead.
    baseline_width_ratio = 1.5 * n_rows
    fig_w = 2.9 * (baseline_width_ratio + n_cols) - 0.95
    fig = plt.figure(figsize=(fig_w, fig_h_total), facecolor='white')
    top_frac = fig_h / fig_h_total
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[baseline_width_ratio] + [1] * n_cols,
                           wspace=0.05, hspace=0.15, top=top_frac, bottom=0.08)

    change_word = '% change' if pct_change else 'change'
    fig.suptitle(f'{var_title}', fontsize=13, fontweight='bold',
                 y=1 - (0.32 / fig_h_total))
    scenario_label = ' vs '.join(s.upper() for s in scenario_order)
    fig.text(0.5, 1 - (0.65 / fig_h_total),
              f'{scenario_label}  |  {change_word} relative to baseline',
              ha='center', fontsize=10.5)

    # ── Baseline panel (spans all rows, own colorbar) ──────────────────────
    ax_base = fig.add_subplot(gs[:, 0], projection=proj)
    ax_base.set_extent(extent, crs=proj)
    _add_background(ax_base, add_satellite, tile_zoom)
    cf_base = ax_base.contourf(lonb, latb, base_smooth, levels=60, cmap=cmap_baseline,
                                transform=proj, extend='both', zorder=3)
    _clip_to_aoi(cf_base, geom_native, ax_base)
    shp_outer.boundary.plot(ax=ax_base, color='black', linewidth=1.2, transform=proj, zorder=6)
    if districts_gdf is not None:
        districts_gdf.boundary.plot(ax=ax_base, color='white', linewidth=0.8, transform=proj, zorder=6)
    ax_base.set_title(f'Baseline\n{unit_baseline}', fontsize=9.5, fontweight='bold')
    gl0 = ax_base.gridlines(draw_labels=True, linewidth=0.3, color='grey', alpha=0.4,
                             linestyle='--', crs=proj)
    gl0.top_labels = False; gl0.right_labels = False
    gl0.xlocator = mticker.MaxNLocator(nbins=4)
    gl0.xlabel_style = {'size': 8}
    cbar_base = fig.colorbar(cf_base, ax=ax_base, location='left', shrink=0.85, pad=0.14, label=unit_baseline)
    cbar_base.set_ticks(_nice_ticks(float(np.nanmin(base_smooth)), float(np.nanmax(base_smooth))))

    # ── Scenario x period grid (shared diverging colorbar) ─────────────────
    grid_axes = []
    cf_grid = None
    for r, scenario in enumerate(scenario_order):
        for c, (tag, label) in enumerate(period_order):
            ax = fig.add_subplot(gs[r, c + 1], projection=proj)
            ax.set_extent(extent, crs=proj)
            grid_axes.append(ax)
            cell = cell_fields.get((scenario, tag))
            _add_background(ax, add_satellite, tile_zoom)
            if cell is not None:
                lon, lat, delta = cell
                # FIX: levels=60 (a bare int) makes contourf auto-compute
                # boundaries from THIS panel's own local data range and
                # ignore vmin/vmax entirely for where those boundaries sit.
                # Since cf_grid gets overwritten every iteration and only
                # the LAST panel's ContourSet is handed to fig.colorbar()
                # below, the shared colorbar ended up showing that one
                # panel's own tiny local range (e.g. "5.136-5.520") instead
                # of the true, intended [vmin, vmax] scale — the fill
                # colors were still correct (they follow vmin/vmax via the
                # norm), only the colorbar's numbers were wrong. Passing an
                # explicit boundary array spanning vmin..vmax fixes this for
                # every panel and makes the colorbar match what's drawn.
                # extend='neither': vmax is now the true max (see above), so
                # nothing actually falls outside [vmin, vmax] anymore.
                cf_grid = ax.contourf(lon, lat, delta, levels=np.linspace(vmin, vmax, 61),
                                       cmap=diverging_cmap, transform=proj, extend='neither', zorder=3)
                _clip_to_aoi(cf_grid, geom_native, ax)
                # Mark cells where pct-change is mathematically undefined
                # (baseline ~0) with hatching instead of leaving them blank —
                # "masked" now means visibly masked, not silently dropped.
                if pct_change:
                    _hatch_masked_region(ax, lon, lat, ~np.isfinite(delta), proj)
            shp_outer.boundary.plot(ax=ax, color='black', linewidth=0.8, transform=proj, zorder=6)
            if districts_gdf is not None:
                districts_gdf.boundary.plot(ax=ax, color='white', linewidth=0.6, transform=proj, zorder=6)
            gl = ax.gridlines(draw_labels=(r == n_rows - 1), linewidth=0.25, color='grey',
                               alpha=0.35, linestyle='--', crs=proj)
            gl.top_labels = False; gl.right_labels = False
            gl.xlocator = mticker.MaxNLocator(nbins=3)
            gl.xlabel_style = {'size': 7}
            if not (r == n_rows - 1):
                gl.bottom_labels = False
            gl.left_labels = False
            if r == 0:
                ax.set_title(label, fontsize=9.5, fontweight='bold')
            if c == 0:
                color = _scenario_color(scenario, scenario_order)
                ax.annotate(scenario.upper(), xy=(-0.14, 0.5), xycoords='axes fraction',
                            rotation=90, va='center', ha='center', fontsize=9,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='none'))

    if cf_grid is not None:
        cbar_grid = fig.colorbar(cf_grid, ax=grid_axes, location='right', shrink=0.85, pad=0.02,
                      label=f'{var_title} {change_word}' + (f' ({unit_delta})' if unit_delta and not pct_change else ''))
        cbar_grid.set_ticks(_nice_ticks(vmin, vmax))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"🗺️  Composite grid map saved ({n_rows}x{n_cols}): {out_path}")


def make_composite_metric_grid(value_grids, scenario_order, period_order,
                                geom_native, districts_gdf, out_path,
                                title_line1, colorbar_label, cmap='viridis',
                                vmin=None, vmax=None, robust_pct=98,
                                extend='neither', add_satellite=True, tile_zoom=8):
    """
    Generic Report-style scenario x period composite grid, NO baseline
    panel — this is what make_composite_agreement_grid now delegates to,
    and what the sensitivity-spread / SNR composites use too (previously
    those two only ever got individual per-cell PNGs, never assembled into
    this grid at all — main.py imported make_composite_agreement_grid but
    had no equivalent call for spread/SNR).

    value_grids : {scenario: {tag: xr.DataArray(lat, lon) or None}}
    vmin/vmax   : pass both for a fixed scale (agreement is naturally
                  0-100%). Leave both None for a data-driven scale spanning
                  the true 0..max of every finite cell in the grid, so no
                  value gets clipped into the "extend" cap color — right for
                  spread/SNR, which are magnitude-only (never negative),
                  unlike the % change grids in make_composite_grid_map which
                  need a symmetric diverging scale.
    extend      : colorbar cap style. 'neither' (default) since the
                  data-driven vmax is the true max, so nothing is clipped.
                  Agreement uses a fixed 20-100 scale where values genuinely
                  fall below 20, so make_composite_agreement_grid passes
                  'both'.
    """
    n_rows, n_cols = len(scenario_order), len(period_order)
    proj = ccrs.PlateCarree()

    shp_outer = gpd.GeoSeries([geom_native], crs='EPSG:4326')
    minx, miny, maxx, maxy = shp_outer.total_bounds
    buf = 0.25
    extent = [minx - buf, maxx + buf, miny - buf, maxy + buf]

    prepped = {}
    all_finite = []
    for scenario in scenario_order:
        for tag, _label in period_order:
            da = value_grids.get(scenario, {}).get(tag)
            if da is None:
                continue
            da_sorted = da.sortby('lat').sortby('lon').transpose('lat', 'lon')
            vals = _smooth_field(da_sorted.values, sigma=1.0)
            prepped[(scenario, tag)] = (da_sorted['lon'].values, da_sorted['lat'].values, vals)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                all_finite.append(finite)

    # FIX: this used to clip vmax to the robust_pct percentile, which could
    # flatten the highest-magnitude cells into the "extend" cap color
    # instead of showing their real value — same issue as
    # make_composite_grid_map's old 98th-percentile clip. Using the true max
    # means every real value in the grid gets its own distinct color.
    if vmin is None or vmax is None:
        vmax = float(np.nanmax(np.concatenate(all_finite))) if all_finite else 1.0
        vmax = vmax or 1.0
        vmin = 0.0

    fig_h = 2.9 * n_rows + 1.4
    title_band_in = 0.9
    fig_h_total = fig_h + title_band_in
    fig = plt.figure(figsize=(1.0 + 2.9 * n_cols, fig_h_total), facecolor='white')
    top_frac = fig_h / fig_h_total
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.05, hspace=0.15, top=top_frac, bottom=0.08)

    fig.suptitle(title_line1, fontsize=13, fontweight='bold', y=1 - (0.32 / fig_h_total))
    scenario_label = '  vs  '.join(s.upper() for s in scenario_order)
    fig.text(0.5, 1 - (0.65 / fig_h_total), scenario_label, ha='center', fontsize=10.5)

    grid_axes, cf_grid = [], None
    for r, scenario in enumerate(scenario_order):
        for c, (tag, label) in enumerate(period_order):
            ax = fig.add_subplot(gs[r, c], projection=proj)
            ax.set_extent(extent, crs=proj)
            grid_axes.append(ax)
            _add_background(ax, add_satellite, tile_zoom)
            cell = prepped.get((scenario, tag))
            if cell is not None:
                lon, lat, vals = cell
                # FIX: same bug as make_composite_grid_map above — levels=60
                # as a bare int ignores vmin/vmax for boundary placement, so
                # the shared colorbar ends up reflecting only the LAST
                # panel's own local data range. Exactly what produced the
                # "1e-11+1e2" degenerate colorbar on the tas model-agreement
                # composite (that field is ~100% almost everywhere, so its
                # last panel's local range was a sliver of a percent wide).
                cf_grid = ax.contourf(lon, lat, vals, levels=np.linspace(vmin, vmax, 61),
                                       cmap=cmap, transform=proj, extend=extend, zorder=3)
                _clip_to_aoi(cf_grid, geom_native, ax)
            shp_outer.boundary.plot(ax=ax, color='black', linewidth=0.8, transform=proj, zorder=6)
            if districts_gdf is not None:
                districts_gdf.boundary.plot(ax=ax, color='white', linewidth=0.6, transform=proj, zorder=6)
            gl = ax.gridlines(draw_labels=(r == n_rows - 1), linewidth=0.25, color='grey',
                               alpha=0.35, linestyle='--', crs=proj)
            gl.top_labels = False; gl.right_labels = False
            gl.xlocator = mticker.MaxNLocator(nbins=3)
            gl.xlabel_style = {'size': 7}
            if not (r == n_rows - 1):
                gl.bottom_labels = False
            gl.left_labels = (c == 0)
            gl.ylabel_style = {'size': 7}
            if r == 0:
                ax.set_title(label, fontsize=9.5, fontweight='bold')
            if c == 0:
                color = _scenario_color(scenario, scenario_order)
                ax.annotate(scenario.upper(), xy=(-0.22, 0.5), xycoords='axes fraction',
                            rotation=90, va='center', ha='center', fontsize=9,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='none'))

    if cf_grid is not None:
        cbar = fig.colorbar(cf_grid, ax=grid_axes, location='right', shrink=0.85, pad=0.02,
                      label=colorbar_label)
        cbar.set_ticks(_nice_ticks(vmin, vmax))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"🗺️  Composite grid map saved ({n_rows}x{n_cols}): {out_path}")


def make_composite_agreement_grid(agreement_grids, scenario_order, period_order,
                                   geom_native, districts_gdf, out_path,
                                   var_title, add_satellite=True, tile_zoom=8,
                                   vmin=20, vmax=100):
    """
    Report-style model-agreement composite — thin wrapper over
    make_composite_metric_grid (fixed 20-100 scale, RdYlGn), kept as its
    own named function since it's what main.py and any existing callers
    already import.
    """
    make_composite_metric_grid(
        agreement_grids, scenario_order, period_order, geom_native, districts_gdf,
        out_path, title_line1=f'{var_title} — Model Agreement — AJK',
        colorbar_label='% of models agreeing on direction', cmap='RdYlGn',
        vmin=vmin, vmax=vmax, extend='both', add_satellite=add_satellite, tile_zoom=tile_zoom)
