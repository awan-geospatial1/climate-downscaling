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

USAGE (paste into a Colab cell, or run as a script in the same environment)
----------------------------------------------------------------------
    JOB_DIR = "/content/drive/MyDrive/climate-downscaling-jobs/job-XXXX"
    exec(open("test_from_cache.py").read())
    # -> gives you: params, ref_cache, corrected_grids, geom_native,
    #    districts_gdf, out_dir, scenarios, future_intervals, models

Then run ANY ONE of the numbered sections at the bottom -- comment out the
ones you don't need. Each section is independent and only needs the objects
built above, not a fresh pipeline run.
"""
import os, json, glob, zipfile
import xarray as xr
from shapely.geometry import mapping

from main import load_shapefile
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
      "districts_gdf, out_dir, scenarios, future_intervals, models\n")

# =============================================================================
# Below: pick ONE section to test. Comment out the rest. Each is independent.
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
    from xclim import indices as xci
    scenario, (start, end, label, tag) = scenarios[0], future_intervals[0]
    model = models[0]
    da = corrected_grids['tas'][scenario][tag][model]
    annual_mean = xci.tg_mean(da, freq='YS').mean(dim='time') - 273.15  # Kelvin -> Celsius
    make_spatial_map(annual_mean, geom_native, '/tmp/test_single_map.png',
                      title=f'tas — {scenario} {tag} {model}', cmap='YlOrRd',
                      add_satellite=params.get('add_satellite_basemap', False))
    print("Saved /tmp/test_single_map.png")

# ── C) Just a composite grid map (tests plot_utils composite/layout changes) ─
if False:
    from plot_utils import make_composite_grid_map
    from xclim import indices as xci
    period_order = [(tag, label) for (s, e, label, tag) in future_intervals]

    def reducer(da):
        return xci.tg_mean(da, freq='YS').mean(dim='time') - 273.15

    baseline_2d = reducer(ref_cache['tas']['ref'])
    scenario_grids = {}
    for scenario in scenarios:
        scenario_grids[scenario] = {}
        for _, _, _, tag in future_intervals:
            per_model = [reducer(da) for da in corrected_grids['tas'][scenario][tag].values()]
            if per_model:
                scenario_grids[scenario][tag] = xr.concat(per_model, dim='model').mean(dim='model')

    make_composite_grid_map(
        baseline_2d, scenario_grids, scenarios, period_order, geom_native, districts_gdf,
        '/tmp/test_composite.png', var_title='Mean Temperature',
        unit_baseline='°C', unit_delta='°C', cmap_baseline='YlOrRd',
        diverging_cmap='RdBu_r', pct_change=False,
        add_satellite=params.get('add_satellite_basemap', False))
    print("Saved /tmp/test_composite.png")

# ── D) Just the Excel report (tests template_excel_utils changes) ──────────
if False:
    from template_excel_utils import write_template_style_excel
    # `results` needs the same shape run_pipeline() builds — cheapest way to
    # get a real one is to run section (A) across all scenario/period combos
    # and assemble `results[f'{scenario}_{tag}']['temperature'/'precipitation']`
    # yourself, or temporarily add a `import pickle; pickle.dump(results, ...)`
    # right after run_pipeline() builds `results`, load it here instead.
    print("See comment above — needs a `results` dict; cheapest source is a "
          "one-time pickle dump from a real run, then reload here.")
