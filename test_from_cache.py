"""
test_from_cache.py — re-run just ONE downstream step against an already-
completed job's cached outputs, without repeating the slow part (GEE fetch
+ QDM bias correction for every model x scenario x period).

WHY THIS WORKS
--------------
run_pipeline() already saves everything the downstream steps need:
  - {job_dir}/request.json                          <- the exact params used
  - {job_dir}/outputs/reference/ref_{var}.nc         <- baseline reference
    (only present if you're using the patched main.py from this
    conversation; if missing, this script fetches it once via GEE and it'll
    be cached for next time too)
  - {job_dir}/outputs/{scenario}/{tag}/data/qdm_{var}_{model}.nc
    <- every bias-corrected model grid

None of the GEE fetching or QDM training has to happen again. This script
rebuilds the exact `corrected_grids` / `ref_cache` structures run_pipeline()
builds internally, then lets you call ONLY the function(s) you're testing.

GENERIC ACROSS VARIABLES / INDICES
-----------------------------------
Every section below is parameterized by IDX_NAME (one of the keys in
main.py's INDEX_MAP: 'annual_mean_tas', 'annual_mean_tasmax',
'annual_mean_tasmin', 'prcptot', 'rx1day') or VAR (raw variable name: 'tas',
'tasmax', 'tasmin', 'pr') for the agreement/sensitivity section. These pull
the matching reducer/units/colormap/pct_change straight from main.py's own
INDEX_MAP / COMPOSITE_CFG / AGREEMENT_COMPOSITE_CFG / SENSITIVITY_UNITS —
the same dicts the real pipeline uses — instead of you hand-copying a
reducer and possibly pairing it with the wrong variable (e.g. a temperature
reducer against precipitation data, which xclim rejects with a unit
mismatch — this is exactly what the old hardcoded version of this script
made easy to do by accident). Just change IDX_NAME / VAR at the top of a
section; everything else follows automatically.

USAGE (paste into a Colab cell, or run as a script in the same environment)
----------------------------------------------------------------------
    JOB_DIR = "/content/drive/MyDrive/climate-downscaling-jobs/job-XXXX"
    exec(open("test_from_cache.py").read())
    # -> gives you: params, ref_cache, corrected_grids, geom_native,
    #    districts_gdf, out_dir, scenarios, future_intervals, models,
    #    period_order, INDEX_MAP, COMPOSITE_CFG, AGREEMENT_COMPOSITE_CFG,
    #    SENSITIVITY_UNITS

Then run ANY ONE of the numbered sections at the bottom — flip its
`if False:` to `if True:`. Each is independent and only needs the objects
built above, not a fresh pipeline run.

Sections: A=indices, B=single spatial map, C=composite grid map,
D=Excel report, E=agreement/spread/SNR composite, F=raw shapefile geometry
overlay (no data — just every polygon, numbered, to sanity-check what's
actually in the AOI shapefile).
"""
import os, json, glob, zipfile
import xarray as xr
from shapely.geometry import mapping

from main import load_shapefile, INDEX_MAP, COMPOSITE_CFG, AGREEMENT_COMPOSITE_CFG, SENSITIVITY_UNITS
from gee_utils import fetch_reference, clean_time_attrs
from config import _CFG

# ── 1) Load params exactly as the original job used them ───────────────────
# NOTE: request.json only has what index.html originally submitted --
# shapefile_path and output_dir are added at RUNTIME by colab_runner.ipynb's
# process_job(), never written back to request.json. Reconstruct both the
# same way it does, or this KeyErrors on 'shapefile_path' immediately.
JOB_DIR = globals().get('JOB_DIR', '/content/drive/MyDrive/climate-downscaling-jobs/REPLACE_ME')
with open(os.path.join(JOB_DIR, 'request.json')) as f:
    params = json.load(f)

out_dir = os.path.join(JOB_DIR, 'outputs')
params['output_dir'] = out_dir


def _resolve_aoi_path(job_dir):
    """Mirrors colab_runner.ipynb's resolve_aoi_path exactly -- see that
    function's docstring for why this can't just assume aoi.geojson."""
    geojson_path = os.path.join(job_dir, 'aoi.geojson')
    zip_path = os.path.join(job_dir, 'aoi.zip')
    if os.path.isfile(geojson_path):
        return geojson_path
    if os.path.isfile(zip_path):
        extract_dir = os.path.join(job_dir, 'aoi_extracted')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        matches = glob.glob(os.path.join(extract_dir, '**', '*.shp'), recursive=True)
        if not matches:
            raise FileNotFoundError(f"aoi.zip found but contained no .shp file (job: {job_dir})")
        return matches[0]
    raise FileNotFoundError(f"Neither aoi.geojson nor aoi.zip found in {job_dir}")


params['shapefile_path'] = _resolve_aoi_path(JOB_DIR)

models = params['models']
scenarios = params['scenarios']
future_intervals = params['future_intervals']  # [[start,end,label,tag], ...]
period_order = [(tag, label) for (start, end, label, tag) in future_intervals]
b_start, b_end = params['baseline_start'], params['baseline_end']
variables = ['tas', 'tasmax', 'tasmin', 'pr']

# ── 2) AOI (fast, local — no GEE needed) ────────────────────────────────────
geom_native, geom_buffered, extent, districts_gdf = load_shapefile(
    params['shapefile_path'], params.get('buffer_km', 25.0))

# ── 3) Reference data — reuse the cache if present, else fetch once ────────
# If this job was run with the reference-caching patch, this is instant.
# If not (an older job folder), this does the ~4-variable GEE fetch once and
# writes the cache so every subsequent test run against this job is instant.
ref_cache = {}
ref_dir = os.path.join(out_dir, 'reference')
os.makedirs(ref_dir, exist_ok=True)
_needs_gee = any(not os.path.exists(os.path.join(ref_dir, f'ref_{v}.nc')) for v in variables)
if _needs_gee:
    import ee
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=params['gee_project_id'])
    region = ee.Geometry(mapping(geom_buffered))

for var in variables:
    ref_path = os.path.join(ref_dir, f'ref_{var}.nc')
    if os.path.exists(ref_path):
        ref_cache[var] = {'ref': clean_time_attrs(xr.open_dataarray(ref_path).load())}
    else:
        ref_cache[var] = fetch_reference(var, b_start, b_end, region, extent, _CFG[var])
        clean_time_attrs(ref_cache[var]['ref']).load().to_netcdf(ref_path)

# ── 4) Corrected model grids — pure disk load, zero GEE calls ──────────────
corrected_grids = {v: {} for v in variables}
missing = []
for var in variables:
    for scenario in scenarios:
        corrected_grids[var][scenario] = {}
        for start, end, label, tag in future_intervals:
            corrected_grids[var][scenario][tag] = {}
            for model in models:
                nc_path = os.path.join(out_dir, scenario, tag, 'data', f'qdm_{var}_{model}.nc')
                if os.path.exists(nc_path):
                    corrected_grids[var][scenario][tag][model] = xr.open_dataarray(nc_path).load()
                else:
                    missing.append(nc_path)

found_count = sum(
    1
    for var in variables
    for scenario in scenarios
    for _, _, _, tag in future_intervals
    for model in models
    if model in corrected_grids[var][scenario][tag]
)
print(f"✅ Loaded corrected grids from cache ({found_count} model/var/scenario/period combos found, {len(missing)} missing)")
if missing:
    print("   Missing (these weren't saved by the original run, or paths differ):")
    for m in missing[:10]:
        print("    -", m)

print("\nReady. Available: params, ref_cache, corrected_grids, geom_native, "
      "districts_gdf, out_dir, scenarios, future_intervals, period_order, models, "
      "INDEX_MAP, COMPOSITE_CFG, AGREEMENT_COMPOSITE_CFG, SENSITIVITY_UNITS\n")
print("INDEX_MAP keys (use any as IDX_NAME below):", list(INDEX_MAP.keys()))
print("Raw variables (use any as VAR in section E):", variables)

# =============================================================================
# Below: pick ONE section to test. Comment out the rest. Each is independent.
# Change IDX_NAME / VAR at the top of a section to switch what it tests --
# everything else (reducer, units, colormap, pct_change) follows from
# INDEX_MAP / COMPOSITE_CFG automatically, so this can't hit the
# "temperature reducer on precipitation data" unit-mismatch error.
# =============================================================================

# ── A) Just indices for one scenario/period (fast — no plotting, no Excel) ──
if False:  # flip to True to run this section
    from indices_utils import compute_temperature_indices, compute_precipitation_indices
    scenario, (start, end, label, tag) = scenarios[0], future_intervals[0]
    model = models[0]
    temp = compute_temperature_indices(
        corrected_grids['tas'][scenario][tag][model],
        corrected_grids['tasmax'][scenario][tag][model],
        corrected_grids['tasmin'][scenario][tag][model],
        start, end, params['temp_thresholds'])
    precip = compute_precipitation_indices(
        corrected_grids['pr'][scenario][tag][model], start, end,
        params['precip_thresholds'], params['wet_months'], params['dry_months'],
        params['return_periods'], n_boot=50,  # small n_boot for a fast test run
        progress_label=f'{model}/{scenario}/{tag}')
    print(temp)
    print(precip['prcptot'], precip['gev_return_levels'])

# ── B) Just a single spatial map (fastest way to test plot_utils changes) ───
if False:
    from plot_utils import make_spatial_map
    IDX_NAME = 'prcptot'  # <- change to any INDEX_MAP key printed above
    var, reducer = INDEX_MAP[IDX_NAME]
    scenario, (start, end, label, tag) = scenarios[0], future_intervals[0]
    model = models[0]
    da = corrected_grids[var][scenario][tag][model]
    field_2d = reducer(da)
    make_spatial_map(field_2d, geom_native, f'{JOB_DIR}/test_single_map_{IDX_NAME}.png',
                      title=f'{IDX_NAME} — {scenario} {tag} {model}',
                      cmap=COMPOSITE_CFG[IDX_NAME]['cmap_baseline'],
                      add_satellite=params.get('add_satellite_basemap', False))
    print(f"Saved {JOB_DIR}/test_single_map_{IDX_NAME}.png")  # in your job's Drive folder, not /tmp

# ── C) Just a composite grid map (tests plot_utils composite/layout changes) ─
if False:
    from plot_utils import make_composite_grid_map
    IDX_NAME = 'rx1day'  # <- change to any INDEX_MAP key printed above
    var, reducer = INDEX_MAP[IDX_NAME]
    cfg_c = COMPOSITE_CFG[IDX_NAME]

    baseline_2d = reducer(ref_cache[var]['ref'])
    scenario_grids = {}
    for scenario in scenarios:
        scenario_grids[scenario] = {}
        for _, _, _, tag in future_intervals:
            per_model = [reducer(da) for da in corrected_grids[var][scenario][tag].values()]
            if per_model:
                scenario_grids[scenario][tag] = xr.concat(per_model, dim='model').mean(dim='model')

    make_composite_grid_map(
        baseline_2d, scenario_grids, scenarios, period_order, geom_native, districts_gdf,
        f'{JOB_DIR}/test_composite_{IDX_NAME}.png',
        var_title=cfg_c['var_title'], unit_baseline=cfg_c['unit_baseline'],
        unit_delta=cfg_c['unit_delta'], cmap_baseline=cfg_c['cmap_baseline'],
        pct_change=cfg_c['pct_change'],
        add_satellite=params.get('add_satellite_basemap', False))
    print(f"Saved {JOB_DIR}/test_composite_{IDX_NAME}.png")

# ── D) Just the Excel report (tests template_excel_utils changes) ──────────
if False:
    from template_excel_utils import write_template_style_excel
    # `results` needs the same shape run_pipeline() builds — cheapest way to
    # get a real one is to run section (A) across all scenario/period combos
    # and assemble `results[f'{scenario}_{tag}']['temperature'/'precipitation']`
    # yourself, or temporarily add `import pickle; pickle.dump(results, open(out_dir+'/results.pkl','wb'))`
    # right after run_pipeline() builds `results`, and reload it here instead:
    #   import pickle
    #   results = pickle.load(open(os.path.join(out_dir, 'results.pkl'), 'rb'))
    print("See comment above — needs a `results` dict; cheapest source is a "
          "one-time pickle dump from a real run, then reload here.")

# ── E) Agreement / spread / SNR composite (tests the 3-way sensitivity fix) ─
if False:
    from plot_utils import make_composite_metric_grid, make_composite_agreement_grid
    from agreement_utils import compute_agreement_and_sensitivity_arrays
    VAR = 'pr'  # <- 'tas', 'tasmax', 'tasmin', or 'pr'
    var_title = AGREEMENT_COMPOSITE_CFG[VAR]
    unit = SENSITIVITY_UNITS.get(VAR, '')

    agreement_grids, spread_grids, snr_grids = ({s: {} for s in scenarios} for _ in range(3))
    for scenario in scenarios:
        for start, end, label, tag in future_intervals:
            grids = corrected_grids[VAR][scenario][tag]
            if len(grids) < 2:
                continue
            agree_da, spread_da, snr_da, used = compute_agreement_and_sensitivity_arrays(
                VAR, scenario, tag, start, end, ref_cache[VAR]['ref'], grids)
            if agree_da is not None:
                agreement_grids[scenario][tag] = agree_da
                spread_grids[scenario][tag] = spread_da
                snr_grids[scenario][tag] = snr_da

    make_composite_agreement_grid(
        agreement_grids, scenarios, period_order, geom_native, districts_gdf,
        f'{JOB_DIR}/test_{VAR}_agreement.png', var_title=var_title,
        add_satellite=params.get('add_satellite_basemap', False))
    make_composite_metric_grid(
        spread_grids, scenarios, period_order, geom_native, districts_gdf,
        f'{JOB_DIR}/test_{VAR}_spread.png',
        title_line1=f'{var_title} — Sensitivity: Inter-Model Spread — AJK',
        colorbar_label=f'Std. dev. across models ({unit})' if unit else 'Std. dev. across models',
        cmap='Purples', add_satellite=params.get('add_satellite_basemap', False))
    print(f"Saved {JOB_DIR}/test_{VAR}_agreement.png and test_{VAR}_spread.png")

# ── F) Raw shapefile overlay — geometry only, no data, no field lookups ────
# Answers "is that uncolored bit at the edge a real district or just a
# contextual boundary (river/neighbor)?" directly, by drawing every polygon
# in the shapefile AS-IS. Deliberately doesn't touch any attribute/name
# column — geopandas doesn't need one to plot geometry, and assuming a
# column name exists (e.g. 'NAME_1') is exactly the kind of guess that
# breaks on a shapefile with a different schema. Only uses what every
# GeoDataFrame always has: .crs, .total_bounds, .geometry, len().
if False:
    import geopandas as gpd
    import matplotlib.pyplot as plt

    raw_gdf = gpd.read_file(params['shapefile_path'])  # unprocessed -- straight from disk
    print("CRS:", raw_gdf.crs)
    print("Extent (minx, miny, maxx, maxy):", raw_gdf.total_bounds)
    print("Polygon count:", len(raw_gdf))
    print("Columns available (FYI only, not required for the plot below):", list(raw_gdf.columns))

    fig, ax = plt.subplots(figsize=(8, 10))
    # One flat color per polygon by row position (cmap index), not by any
    # field value -- works identically regardless of schema.
    raw_gdf.plot(ax=ax, cmap='tab20', edgecolor='black', linewidth=0.6, alpha=0.85)
    for i, geom in enumerate(raw_gdf.geometry):
        c = geom.representative_point()
        ax.annotate(str(i), (c.x, c.y), fontsize=8, ha='center', fontweight='bold')
    ax.set_title(f'Raw shapefile — {len(raw_gdf)} polygon(s), CRS={raw_gdf.crs}')
    fig.savefig(f'{JOB_DIR}/test_raw_shapefile_overlay.png', dpi=130, bbox_inches='tight')
    print(f"Saved {JOB_DIR}/test_raw_shapefile_overlay.png — each polygon numbered by its "
          f"row index, so you can match the shape you're curious about to whatever row it is, "
          f"then look up that one row's attributes if you still want them (df.iloc[i]) —"
          f"but the shape itself is already visible without needing to.")
