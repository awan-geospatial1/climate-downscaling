import os, traceback, pandas as pd, numpy as np, xarray as xr, geopandas as gpd
from shapely.geometry import mapping

from config import _CFG, DEFAULT_NQUANTILES, DEFAULT_QDM_GROUP, DEFAULT_WET_THRESH, DEFAULT_CHUNKS_LATLON
from gee_utils import fetch_reference, fetch_cmip6, regrid_to_reference, clean_time_attrs
from qdm_utils import train_qdm, apply_qdm, adjust_wet_day_frequency
from indices_utils import (compute_temperature_indices, compute_precipitation_indices, aggregate_across_models,
                           daily_spatial_series, daily_spatial_ensemble)
from plot_utils import (plot_fan_chart, make_spatial_map, plot_index_comparison,
                         make_composite_grid_map, make_composite_agreement_grid,
                         make_composite_metric_grid)
from ensemble_utils import compute_ensemble_mean, compute_ensemble_max, save_ensemble_netcdf
from agreement_utils import make_agreement_sensitivity_maps
from template_excel_utils import write_template_style_excel
from xclim import indices as xci

# ── Shared spatial-map / composite-map config ───────────────────────────────
# Pulled out to module level (instead of being defined inline inside
# run_pipeline) so anything outside a live pipeline run — test_from_cache.py
# in particular — can import the SAME idx_name -> (var, reducer) mapping and
# the SAME idx_name -> (units, colormap, pct_change) config the real
# pipeline uses, rather than hand-copying them and risking exactly the kind
# of mismatch that's easy to introduce by hand (e.g. pairing a temperature
# reducer like xci.tg_mean with precipitation data, which xclim correctly
# rejects with a unit-validation error). One definition, two consumers.
#
# NOTE: reducers here must return a 2D (lat, lon) DataArray. The -273.15 on
# every temperature reducer converts Kelvin (this pipeline's internal unit
# throughout, per config.py's ref_units='K') to Celsius for DISPLAY only —
# every delta/change computation subtracts two of these, so the constant
# per-field offset cancels out and isn't affected either way.
INDEX_MAP = {
    'annual_mean_tas': ('tas', lambda da: xci.tg_mean(da, freq='YS').mean(dim='time') - 273.15),
    'prcptot': ('pr', lambda da: xci.precip_accumulation(da, freq='YS').mean(dim='time')),
    'annual_mean_tasmax': ('tasmax', lambda da: xci.tx_mean(da, freq='YS').mean(dim='time') - 273.15),
    'annual_mean_tasmin': ('tasmin', lambda da: xci.tn_mean(da, freq='YS').mean(dim='time') - 273.15),
    'rx1day': ('pr', lambda da: xci.max_1day_precipitation_amount(da, freq='YS').mean(dim='time')),
}

COMPOSITE_CFG = {
    'annual_mean_tas': dict(pct_change=False, cmap_baseline='YlOrRd',
                             var_title='Mean Temperature', unit_baseline='°C', unit_delta='°C'),
    'annual_mean_tasmax': dict(pct_change=False, cmap_baseline='YlOrRd',
                                var_title='Mean Max Temperature', unit_baseline='°C', unit_delta='°C'),
    'annual_mean_tasmin': dict(pct_change=False, cmap_baseline='YlOrRd',
                                var_title='Mean Min Temperature', unit_baseline='°C', unit_delta='°C'),
    'prcptot': dict(pct_change=True, cmap_baseline='YlGnBu',
                     var_title='Precipitation', unit_baseline='mm/year', unit_delta='%'),
    'rx1day': dict(pct_change=True, cmap_baseline='YlGnBu',
                    var_title='Max 1-Day Precipitation', unit_baseline='mm', unit_delta='%'),
}

# Agreement/sensitivity composites are keyed by raw variable name (tas, pr),
# not idx_name, since agreement/spread/SNR are computed once per variable
# (not once per derived index like INDEX_MAP above).
AGREEMENT_COMPOSITE_CFG = {'tas': 'Mean Temperature', 'pr': 'Precipitation'}
SENSITIVITY_UNITS = {'tas': '°C', 'tasmax': '°C', 'tasmin': '°C', 'pr': '% points'}


def load_shapefile(shp_path, buffer_km):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    geom_native = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    buffered_utm = gdf_utm.buffer(buffer_km * 1000.0)
    geom_buffered = (gpd.GeoSeries(buffered_utm, crs=utm_crs)
                      .to_crs(epsg=4326)
                      .union_all() if hasattr(gpd.GeoSeries, 'union_all')
                      else gpd.GeoSeries(buffered_utm, crs=utm_crs).to_crs(epsg=4326).unary_union)
    minx, miny, maxx, maxy = geom_buffered.bounds
    extent = [minx, miny, maxx, maxy]
    # `gdf` (pre-dissolve) is returned too so composite grid maps can draw
    # each district/sub-basin's own boundary, not just the outer AOI edge —
    # union_all() above is what a single-polygon caller (AOI clip/buffer)
    # needs, but it throws away exactly the internal boundaries the
    # AJK-style composite maps want.
    return geom_native, geom_buffered, extent, gdf


def run_pipeline(params):
    import ee

    shp_path = params['shapefile_path']
    buffer_km = params.get('buffer_km', 25.0)
    gee_project_id = params['gee_project_id']
    models = params['models']
    scenarios = params['scenarios']
    b_start = params['baseline_start']
    b_end = params['baseline_end']
    h_start = params.get('hist_start', b_start)
    future_intervals = params['future_intervals']
    wet_months = params['wet_months']
    dry_months = params['dry_months']
    temp_thresholds = params['temp_thresholds']
    precip_thresholds = params['precip_thresholds']
    return_periods = params['return_periods']
    n_boot = params.get('gev_n_bootstrap', 1000)
    nquantiles = params.get('nquantiles', DEFAULT_NQUANTILES)
    qdm_group = params.get('qdm_group', DEFAULT_QDM_GROUP)
    wet_thresh = params.get('wet_thresh', DEFAULT_WET_THRESH)
    chunks = params.get('chunks_latlon', DEFAULT_CHUNKS_LATLON)
    out_dir = params['output_dir']
    add_satellite = params.get('add_satellite_basemap', False)
    # FIX: every composite map title had "AJK" hard-coded directly in
    # plot_utils.py/main.py f-strings — fine for this one region, wrong the
    # moment this pipeline runs against anywhere else. index.html's job
    # submission form doesn't send this field yet (nothing to send), so the
    # 'AJK' default here preserves today's behavior exactly until that's
    # added; passing region_name explicitly in params overrides it.
    region_name = params.get('region_name', 'AJK')

    headline_stat = params.get('headline_stat', {
        'annual_mean_tas': 'mean', 'annual_mean_tasmax': 'mean',
        'annual_mean_tasmin': 'mean', 'monthly_mean_tas': 'mean',
        'su_days_per_month': 'mean', 'prcptot': 'mean',
        'wet_season_total': 'mean', 'dry_season_total': 'mean',
        'rx1day': 'p90', 'rx5day': 'p90',
        'wetdays_per_month': 'mean', 'gev_return_level': 'p90',
    })

    # FIX (new capability, not a bug fix): the interactive ee.Authenticate()
    # branch below opens a browser OAuth prompt -- fine for a person running
    # cells by hand, but a scheduled/unattended Colab run has nobody there
    # to click through it, and the run will just hang/fail. If the caller
    # supplies a service account (params['gee_service_account'] +
    # params['gee_key_data'], the JSON key's contents as a string), use
    # that non-interactive path instead. Existing interactive callers are
    # unaffected -- these two params are optional and default to None.
    gee_service_account = params.get('gee_service_account')
    gee_key_data = params.get('gee_key_data')

    if gee_service_account and gee_key_data:
        credentials = ee.ServiceAccountCredentials(gee_service_account, key_data=gee_key_data)
        ee.Initialize(credentials=credentials, project=gee_project_id)
        print(f"✅ GEE initialised via service account: {gee_service_account}")
    else:
        try:
            ee.Initialize()
            print("✅ GEE already initialised.")
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=gee_project_id)
            print(f"✅ GEE initialised with project: {gee_project_id}")

    geom_native, geom_buffered, extent, districts_gdf = load_shapefile(shp_path, buffer_km)
    region = ee.Geometry(mapping(geom_buffered))
    print("✅ AOI prepared.")

    os.makedirs(out_dir, exist_ok=True)

    # ── Output folder layout ────────────────────────────────────────────
    # out_dir/
    #   Baseline/tables/
    #   <scenario>/<tag>/data/     ← corrected netCDF grids
    #   <scenario>/<tag>/maps/     ← spatial maps for that scenario+period
    #   <scenario>/tables/         ← per-scenario CSV summary
    #   tables/                    ← master Excel summary (all scenarios/periods)
    #   graphs/                   ← fan charts + scenario-comparison bar charts
    tables_dir = os.path.join(out_dir, 'tables')
    graphs_dir = os.path.join(out_dir, 'graphs')
    baseline_tables_dir = os.path.join(out_dir, 'Baseline', 'tables')
    for d in (tables_dir, graphs_dir, baseline_tables_dir):
        os.makedirs(d, exist_ok=True)

    def scenario_period_dirs(scenario, tag):
        data_dir = os.path.join(out_dir, scenario, tag, 'data')
        maps_dir = os.path.join(out_dir, scenario, tag, 'maps')
        ensemble_dir = os.path.join(out_dir, scenario, tag, 'ensemble')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(maps_dir, exist_ok=True)
        os.makedirs(ensemble_dir, exist_ok=True)
        return data_dir, maps_dir, ensemble_dir

    def scenario_tables_dir(scenario):
        d = os.path.join(out_dir, scenario, 'tables')
        os.makedirs(d, exist_ok=True)
        return d

    variables = ['tas', 'tasmax', 'tasmin', 'pr']
    for v in variables:
        _CFG[v]['chunks'] = chunks

    ref_cache = {}
    ref_dir = os.path.join(out_dir, 'reference')
    os.makedirs(ref_dir, exist_ok=True)
    for var in variables:
        # FIX: fetch_reference() was called unconditionally every run, with
        # no caching -- unlike the corrected CMIP6 grids just below (which
        # get saved to data_dir and can be reloaded), the reference dataset
        # had to be re-fetched from GEE from scratch every single time, even
        # when only testing/iterating on a downstream step (indices, Excel,
        # plotting) that doesn't touch the reference data at all. Caching it
        # to {out_dir}/reference/ref_{var}.nc means a second run against the
        # same AOI/baseline period/output folder skips this fetch entirely --
        # see test_from_cache.py for a standalone script that reuses this
        # plus the already-cached corrected_grids .nc files to test any
        # downstream step without touching GEE at all.
        ref_path = os.path.join(ref_dir, f'ref_{var}.nc')
        if os.path.exists(ref_path):
            print(f"📂 Using cached reference for {var} → {ref_path}")
            ref_cache[var] = {'ref': clean_time_attrs(xr.open_dataarray(ref_path).load())}
        else:
            ref_cache[var] = fetch_reference(var, b_start, b_end, region, extent, _CFG[var])
            try:
                clean_time_attrs(ref_cache[var]['ref']).load().to_netcdf(ref_path)
                print(f"💾 Reference for {var} cached → {ref_path}")
            except Exception as e:
                print(f"⚠️ Could not cache reference for {var} ({e}); will re-fetch next run")

    qdm_cache = {v: {} for v in variables}
    hist_cache = {v: {} for v in variables}
    for var in variables:
        cfg = _CFG[var]
        for model in models:
            try:
                ds_calib = fetch_cmip6(var, model, 'historical', b_start, b_end, region, extent, cfg,
                                        f'{var}/{model}/hist-calib')
                da_calib = regrid_to_reference(ds_calib[var], ref_cache[var]['ref'], cfg['fill_na'])
                qdm = train_qdm(ref_cache[var]['ref'], da_calib, cfg['qdm_kind'], nquantiles, qdm_group)
                qdm_cache[var][model] = qdm

                ds_full = fetch_cmip6(var, model, 'historical', h_start, b_end, region, extent, cfg,
                                       f'{var}/{model}/hist-full')
                da_full = regrid_to_reference(ds_full[var], ref_cache[var]['ref'], cfg['fill_na'])
                corr = apply_qdm(qdm, da_full, cfg['clip_min'], cfg['clip_max'], cfg['ref_units'])
                if cfg['wet_adjust']:
                    corr = adjust_wet_day_frequency(ref_cache[var]['ref'], corr, thresh=wet_thresh)
                hist_cache[var][model] = corr
            except Exception as e:
                print(f"⚠️ {var}/{model} QDM failed: {e}")
                traceback.print_exc()

    corrected_grids = {v: {} for v in variables}
    for var in variables:
        cfg = _CFG[var]
        for scenario in scenarios:
            corrected_grids[var][scenario] = {}
            for start, end, label, tag in future_intervals:
                corrected_grids[var][scenario][tag] = {}
                for model in models:
                    if model not in qdm_cache[var]:
                        continue
                    try:
                        ds = fetch_cmip6(var, model, scenario, start, end, region, extent, cfg,
                                          f'{var}/{model}/{scenario}/{tag}')
                        da = regrid_to_reference(ds[var], ref_cache[var]['ref'], cfg['fill_na'])
                        corr = apply_qdm(qdm_cache[var][model], da, cfg['clip_min'], cfg['clip_max'], cfg['ref_units'])
                        if cfg['wet_adjust']:
                            corr = adjust_wet_day_frequency(ref_cache[var]['ref'], corr, thresh=wet_thresh)
                        corr = clean_time_attrs(corr)
                        data_dir, _, _ = scenario_period_dirs(scenario, tag)
                        out_path = os.path.join(data_dir, f'qdm_{var}_{model}.nc')
                        corr.load().to_netcdf(out_path)
                        corrected_grids[var][scenario][tag][model] = corr
                        print(f"💾 {var}/{model}/{scenario}/{tag} saved → {out_path}")
                    except Exception as e:
                        print(f"⚠️ {var}/{model}/{scenario}/{tag} failed: {e}")
                        traceback.print_exc()

                # ── Ensemble mean (every variable) + ensemble max (pr only) ──
                # Previously the pipeline never persisted a general
                # per-variable ensemble grid -- only two hardcoded spatial-
                # map indices got an in-memory, throwaway ensemble mean.
                # This saves an actual ensemble_mean_<var>.nc for every
                # variable/scenario/period (matching what the README already
                # documented as part of the output layout, but main.py never
                # actually produced), plus an ensemble_max_pr.nc for
                # precipitation specifically -- the elementwise maximum
                # across models at every cell/day, useful for extreme-value
                # / hazard-style analysis (see Max_Precip / Return Period
                # Graphs in the Excel report below), which a mean grid isn't
                # meant for.
                grids = corrected_grids[var][scenario][tag]
                if grids:
                    _, _, ensemble_dir = scenario_period_dirs(scenario, tag)
                    try:
                        ens_mean = compute_ensemble_mean(grids)
                        mean_path = os.path.join(ensemble_dir, f'ensemble_mean_{var}.nc')
                        save_ensemble_netcdf(ens_mean, mean_path, units=cfg['ref_units'])
                        print(f"📊 Ensemble mean saved ({len(grids)} model(s)) → {mean_path}")
                    except Exception as e:
                        print(f"⚠️ Ensemble mean for {var}/{scenario}/{tag} failed: {e}")
                        traceback.print_exc()

                    if var == 'pr':
                        try:
                            ens_max = compute_ensemble_max(grids)
                            max_path = os.path.join(ensemble_dir, f'ensemble_max_{var}.nc')
                            save_ensemble_netcdf(ens_max, max_path, units=cfg['ref_units'])
                            print(f"📊 Ensemble max saved ({len(grids)} model(s)) → {max_path}")
                        except Exception as e:
                            print(f"⚠️ Ensemble max for {var}/{scenario}/{tag} failed: {e}")
                            traceback.print_exc()
                else:
                    print(f"⚠️ No models succeeded for {var}/{scenario}/{tag} — "
                          f"skipping ensemble mean/max (nothing to aggregate).")

    results = {}
    print("\n📈 Computing baseline indices (incl. GEV return-period bootstrap)...")
    temp_baseline = compute_temperature_indices(ref_cache['tas']['ref'], ref_cache['tasmax']['ref'],
                                                 ref_cache['tasmin']['ref'], b_start, b_end, temp_thresholds)
    precip_baseline = compute_precipitation_indices(ref_cache['pr']['ref'], b_start, b_end,
                                                      precip_thresholds, wet_months, dry_months,
                                                      return_periods, n_boot, progress_label='Baseline')
    print("✅ Baseline indices computed.")

    # FIX (pre-existing bug, not introduced by anything above): this used to
    # wrap SCALAR baseline values into {'mean','p10','p90'} but leave
    # LIST-valued ones (monthly_mean_tas, wetdays_per_month_*mm,
    # su_days_per_month_*C) as raw lists. Every scenario/period gets those
    # same list-valued indices wrapped into {'mean','p10','p90'} dicts by
    # aggregate_across_models() -- so Baseline was the only period with a
    # different shape for the same index. The rows-building loop below
    # unconditionally calls `stats.get('mean')`, which crashes with
    # AttributeError on a raw list -- this reliably crashed on every real
    # run, since monthly_mean_tas is always present. Only gev_return_levels
    # (itself a dict, keyed by return period) should stay unwrapped, since
    # the rows loop already special-cases that one by its raw shape.
    def _wrap_baseline_index(v):
        return v if isinstance(v, dict) else {'mean': v, 'p10': v, 'p90': v}

    results['Baseline'] = {
        'temperature': {k: _wrap_baseline_index(v) for k, v in temp_baseline.items()},
        'precipitation': {k: _wrap_baseline_index(v) for k, v in precip_baseline.items()}
    }

    for scenario in scenarios:
        for start, end, label, tag in future_intervals:
            key = f'{scenario}_{tag}'
            print(f"📈 Computing indices for {scenario}/{tag} (incl. GEV return-period bootstrap, "
                  f"{n_boot} iterations/model)...")
            temp_list, precip_list = [], []
            for model in models:
                if (model not in corrected_grids['tas'][scenario][tag] or
                        model not in corrected_grids['tasmax'][scenario][tag] or
                        model not in corrected_grids['tasmin'][scenario][tag] or
                        model not in corrected_grids['pr'][scenario][tag]):
                    continue
                tas = corrected_grids['tas'][scenario][tag][model]
                tasmax = corrected_grids['tasmax'][scenario][tag][model]
                tasmin = corrected_grids['tasmin'][scenario][tag][model]
                pr = corrected_grids['pr'][scenario][tag][model]
                temp_list.append(compute_temperature_indices(tas, tasmax, tasmin, start, end, temp_thresholds))
                precip_list.append(compute_precipitation_indices(pr, start, end, precip_thresholds,
                                                                   wet_months, dry_months, return_periods, n_boot,
                                                                   progress_label=f'{model}/{scenario}/{tag}'))
                print(f"   ✅ {model} indices done ({len(temp_list)}/{len(models)})")
            results[key] = {
                'temperature': aggregate_across_models(temp_list, return_periods),
                'precipitation': aggregate_across_models(precip_list, return_periods)
            }
    print("✅ All scenario/period indices computed.\n")

    rows = []
    for period, groups in results.items():
        for domain, idx_dict in groups.items():
            for idx_name, stats in idx_dict.items():
                if idx_name == 'gev_return_levels':
                    for T, s in stats.items():
                        headline = s.get(headline_stat.get('gev_return_level', 'p90'), s['mean'])
                        rows.append([period, domain, f'{idx_name}_{T}yr', s['mean'], s['p10'], s['p90'], headline])
                elif isinstance(stats.get('mean'), list):
                    for m in range(12):
                        base = idx_name.rsplit('_', 1)[0] if idx_name[-1].isdigit() else idx_name
                        hstat = headline_stat.get(base, 'mean')
                        headline = stats[hstat][m]
                        rows.append([period, domain, f'{idx_name}_m{m+1:02d}', stats['mean'][m], stats['p10'][m], stats['p90'][m], headline])
                else:
                    base = idx_name.split('_mean')[0].split('_p90')[0]
                    hstat = headline_stat.get(base, 'mean')
                    headline = stats.get(hstat, stats['mean'])
                    rows.append([period, domain, idx_name, stats['mean'], stats['p10'], stats['p90'], headline])

    df = pd.DataFrame(rows, columns=['period', 'domain', 'index', 'mean', 'p10', 'p90', 'headline_value'])
    excel_path = os.path.join(tables_dir, 'climate_indices_summary.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"✅ Master Excel summary saved: {excel_path}")

    # Per-scenario CSV (all periods for that scenario) + a Baseline CSV
    baseline_csv = os.path.join(baseline_tables_dir, 'baseline_summary.csv')
    df[df['period'] == 'Baseline'].to_csv(baseline_csv, index=False)
    print(f"✅ Baseline table saved: {baseline_csv}")
    for scenario in scenarios:
        scen_df = df[df['period'].str.startswith(f'{scenario}_')]
        if scen_df.empty:
            continue
        scen_csv = os.path.join(scenario_tables_dir(scenario), f'{scenario}_summary.csv')
        scen_df.to_csv(scen_csv, index=False)
        print(f"✅ {scenario} table saved: {scen_csv}")

    # ── Daily spatial-average Excel ─────────────────────────────────────
    # One sheet per period (Baseline obs, historical bias-corrected ensemble,
    # and each scenario/future-period), each with one column per variable:
    # the daily spatial (area) mean across the AOI, ensemble-averaged across
    # models where relevant. Temperatures are converted K → °C for readability.
    daily_path = os.path.join(tables_dir, 'daily_spatial_averages.xlsx')
    sheets = {}

    base_cols = {}
    for var in variables:
        try:
            base_cols[var] = daily_spatial_series(ref_cache[var]['ref'], _CFG[var]['ref_units'])
        except Exception as e:
            print(f"⚠️ daily baseline series for {var} failed: {e}")
    if base_cols:
        sheets['Baseline_obs'] = pd.DataFrame(base_cols)

    hist_cols = {}
    for var in variables:
        try:
            s = daily_spatial_ensemble(hist_cache[var], _CFG[var]['ref_units'])
            if s is not None:
                hist_cols[var] = s
        except Exception as e:
            print(f"⚠️ daily historical-corrected series for {var} failed: {e}")
    if hist_cols:
        sheets['Historical_corrected'] = pd.DataFrame(hist_cols)

    for scenario in scenarios:
        for start, end, label, tag in future_intervals:
            cols = {}
            for var in variables:
                try:
                    s = daily_spatial_ensemble(corrected_grids[var][scenario][tag], _CFG[var]['ref_units'])
                    if s is not None:
                        cols[var] = s
                except Exception as e:
                    print(f"⚠️ daily series for {var}/{scenario}/{tag} failed: {e}")
            if cols:
                sheets[f'{scenario}_{tag}'[:31]] = pd.DataFrame(cols)  # Excel sheet-name length limit

    if sheets:
        try:
            with pd.ExcelWriter(daily_path, engine='openpyxl') as writer:
                for sheet_name, sheet_df in sheets.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name)
            print(f"✅ Daily spatial-average Excel saved ({len(sheets)} sheet(s)): {daily_path}")
        except Exception as e:
            print(f"⚠️ Daily spatial-average Excel failed: {e}")
            traceback.print_exc()
    else:
        print("❌ Daily spatial-average Excel not written — no data available for any period.")

    # ── Template-style Excel workbook ────────────────────────────────────
    # A second workbook matching the layout requested (Daily Spatial
    # Averages / Helper / Precipitation Stats and Graph / Max_Precip /
    # Return Period Graphs / Temperature Stats and Graphs), built from the
    # same corrected_grids/results this pipeline already computes. See
    # template_excel_utils.py's module docstring for the two judgment
    # calls made in there (ensemble-max definition, Gumbel vs. GEV) and
    # what this version does and doesn't reproduce pixel-for-pixel from
    # the original template (short version: all 6 sheets' DATA is real
    # and correct; only 2 of the original's 13 native charts are included
    # so far).
    try:
        template_path = os.path.join(tables_dir, 'template_style_report.xlsx')
        write_template_style_excel(template_path, variables, _CFG, scenarios, future_intervals,
                                    ref_cache, corrected_grids, results, precip_thresholds, return_periods)
        print(f"✅ Template-style Excel report saved: {template_path}")
    except Exception as e:
        print(f"⚠️ Template-style Excel report failed: {e}")
        traceback.print_exc()

    try:
        plot_fan_chart(hist_cache, corrected_grids, 'tas', 'Annual mean temperature (K)',
                       os.path.join(graphs_dir, 'fanchart_tas.png'), 'Temperature',
                       scenarios, future_intervals, models)
    except Exception as e:
        print(f"⚠️ fanchart_tas.png failed: {e}")
        traceback.print_exc()
    try:
        plot_fan_chart(hist_cache, corrected_grids, 'pr', 'Annual total precipitation (mm)',
                       os.path.join(graphs_dir, 'fanchart_pr.png'), 'Precipitation',
                       scenarios, future_intervals, models)
    except Exception as e:
        print(f"⚠️ fanchart_pr.png failed: {e}")
        traceback.print_exc()
    for fname in ('fanchart_tas.png', 'fanchart_pr.png'):
        fpath = os.path.join(graphs_dir, fname)
        print(f"{'✅' if os.path.exists(fpath) else '❌'} {fname} {'exists' if os.path.exists(fpath) else 'was NOT created'}")

    # ── Scenario-comparison bar charts ──────────────────────────────────
    comparison_indices = ['annual_mean_tas', 'annual_mean_tasmax', 'annual_mean_tasmin',
                           'prcptot', 'wet_season_total', 'dry_season_total']
    charts_made = 0
    for idx_name in comparison_indices:
        try:
            if plot_index_comparison(df, idx_name, os.path.join(graphs_dir, f'compare_{idx_name}.png')):
                charts_made += 1
        except Exception as e:
            print(f"⚠️ compare_{idx_name}.png failed: {e}")
            traceback.print_exc()
    print(f"✅ {charts_made} scenario-comparison bar charts saved → {graphs_dir}")

    # ── Spatial maps ────────────────────────────────────────────────────
    # NOTE: reducers here must return a 2D (lat, lon) DataArray.
    index_map = INDEX_MAP

    maps_made = 0
    maps_skipped = []
    for idx_name, (var, reducer) in index_map.items():
        for scenario in scenarios:
            for start, end, label, tag in future_intervals:
                grids = corrected_grids.get(var, {}).get(scenario, {}).get(tag, {})
                if not grids:
                    maps_skipped.append(f'{idx_name}/{scenario}/{tag} (no corrected grids — check ⚠️ QDM/fetch warnings above)')
                    continue
                try:
                    per_model = [reducer(da) for da in grids.values()]
                    ens_mean = xr.concat(per_model, dim='model').mean(dim='model')
                    _, maps_dir, _ = scenario_period_dirs(scenario, tag)
                    out_path = os.path.join(maps_dir, f'{idx_name}_{scenario}_{tag}.png')
                    make_spatial_map(ens_mean, geom_native, out_path,
                                      title=f'{idx_name} – {scenario.upper()} ({tag})',
                                      add_satellite=add_satellite)
                    maps_made += 1
                except Exception as e:
                    maps_skipped.append(f'{idx_name}/{scenario}/{tag} (map generation failed: {e})')
                    traceback.print_exc()

    if maps_made:
        print(f"✅ Spatial maps saved: {maps_made}")
    else:
        print("❌ No spatial maps were generated.")
    for reason in maps_skipped:
        print(f"   ⚠️ skipped: {reason}")

    # ── Composite AJK-style grid maps (temperature, precipitation, extremes) ─
    # Baseline panel + a scenario x period grid sharing one colorbar, per
    # index. scenario_order/period_order come straight from this run's own
    # params, so the grid auto-sizes to however many scenarios (3, 4, ...)
    # and periods (3, 4, 5, ...) were actually selected — no code change
    # needed for a bigger run.
    period_order = [(tag, label) for (start, end, label, tag) in future_intervals]
    composite_cfg = COMPOSITE_CFG
    composite_maps_made = 0
    for idx_name, (var, reducer) in index_map.items():
        cfg_c = composite_cfg.get(idx_name)
        if cfg_c is None:
            continue
        try:
            baseline_ref = ref_cache.get(var, {}).get('ref')
            if baseline_ref is None:
                print(f"   ⚠️ composite grid skipped for {idx_name}: no baseline reference field")
                continue
            baseline_2d = reducer(baseline_ref)

            scenario_grids_2d = {}
            for scenario in scenarios:
                scenario_grids_2d[scenario] = {}
                for start, end, label, tag in future_intervals:
                    grids = corrected_grids.get(var, {}).get(scenario, {}).get(tag, {})
                    if not grids:
                        continue
                    per_model = [reducer(da) for da in grids.values()]
                    scenario_grids_2d[scenario][tag] = xr.concat(per_model, dim='model').mean(dim='model')

            composite_path = os.path.join(graphs_dir, f'composite_{idx_name}.png')
            make_composite_grid_map(
                baseline_2d, scenario_grids_2d, scenarios, period_order,
                geom_native, districts_gdf, composite_path,
                region_name=region_name, add_satellite=add_satellite, **cfg_c)
            composite_maps_made += 1
        except Exception as e:
            print(f"⚠️ composite grid map for {idx_name} failed: {e}")
            traceback.print_exc()
    print(f"✅ Composite grid maps saved: {composite_maps_made}")

    # ── Model agreement & sensitivity maps ──────────────────────────────
    # For every variable/scenario/period: % of models agreeing on the
    # direction of change vs baseline, inter-model spread, and
    # signal-to-noise ratio. See agreement_utils.py's module docstring for
    # what these mean; see the earlier AJK conversation for the fuller
    # explanation this was originally built against.
    #
    # FIX: this used to only ever assemble a composite scenario x period
    # GRID for model_agreement (via make_composite_agreement_grid) — spread
    # and SNR got individual per-scenario/period PNGs but were never
    # collected into the matching composite grid at all, even though that's
    # exactly the report figure (e.g. pr_sensitivity_spread.png) the pipeline
    # is meant to produce. Also switched to the single-pass
    # compute_agreement_and_sensitivity_arrays so the per-model change stack
    # is computed once per var/scenario/period instead of twice (previously
    # make_agreement_sensitivity_maps and compute_agreement_array each
    # recomputed it independently for the same cell).
    from agreement_utils import compute_agreement_and_sensitivity_arrays
    agreement_maps_made = 0
    agreement_composite_cfg = AGREEMENT_COMPOSITE_CFG
    sensitivity_units = SENSITIVITY_UNITS
    for var in variables:
        agreement_grids_2d = {s: {} for s in scenarios}
        spread_grids_2d = {s: {} for s in scenarios}
        snr_grids_2d = {s: {} for s in scenarios}
        for scenario in scenarios:
            for start, end, label, tag in future_intervals:
                grids = corrected_grids.get(var, {}).get(scenario, {}).get(tag, {})
                if len(grids) < 2:
                    print(f"   ⚠️ agreement/sensitivity skipped for {var}/{scenario}/{tag}: "
                          f"need >= 2 models, have {len(grids)}.")
                    continue
                try:
                    _, maps_dir, _ = scenario_period_dirs(scenario, tag)
                    agree_da, spread_da, snr_da, used_models = compute_agreement_and_sensitivity_arrays(
                        var, scenario, tag, start, end, ref_cache[var]['ref'], grids)
                    if agree_da is None:
                        print(f"    [ERROR] Only {len(used_models)} usable model(s) for "
                              f"{var}/{scenario}/{tag} agreement/sensitivity; need >= 2, skipping.")
                        continue

                    specs = [
                        ('model_agreement', agree_da, 'RdYlGn', '% of models agreeing on direction of change'),
                        ('sensitivity_spread', spread_da, 'Purples', 'Inter-model spread (std. dev. of change)'),
                        ('sensitivity_snr', snr_da, 'YlOrBr', 'Signal-to-noise ratio (|mean change| / spread)'),
                    ]
                    for suffix, da_2d, cmap, subtitle in specs:
                        try:
                            out_path = f'{maps_dir}/{var}_{suffix}_{scenario}_{tag}.png'
                            make_spatial_map(da_2d, geom_native, out_path,
                                              title=f'{var} {subtitle} \u2014 {scenario.upper()} ({tag})',
                                              cmap=cmap, add_satellite=add_satellite)
                            agreement_maps_made += 1
                        except Exception as e:
                            print(f"    [ERROR] {var}/{scenario}/{tag} {suffix} map failed: {e}")
                    print(f"    used {len(used_models)} model(s) for agreement/sensitivity: {used_models}")

                    agreement_grids_2d[scenario][tag] = agree_da
                    spread_grids_2d[scenario][tag] = spread_da
                    snr_grids_2d[scenario][tag] = snr_da
                except Exception as e:
                    print(f"⚠️ agreement/sensitivity maps for {var}/{scenario}/{tag} failed: {e}")
                    traceback.print_exc()

        if var in agreement_composite_cfg:
            var_title = agreement_composite_cfg[var]
            unit = sensitivity_units.get(var, '')
            try:
                make_composite_agreement_grid(
                    agreement_grids_2d, scenarios, period_order,
                    geom_native, districts_gdf,
                    os.path.join(graphs_dir, f'composite_{var}_model_agreement.png'),
                    var_title=var_title, region_name=region_name, add_satellite=add_satellite)
            except Exception as e:
                print(f"⚠️ composite agreement grid for {var} failed: {e}")
                traceback.print_exc()
            try:
                make_composite_metric_grid(
                    spread_grids_2d, scenarios, period_order,
                    geom_native, districts_gdf,
                    os.path.join(graphs_dir, f'composite_{var}_sensitivity_spread.png'),
                    title_line1=f'{var_title} \u2014 Sensitivity: Inter-Model Spread \u2014 {region_name}',
                    colorbar_label=f'Std. dev. across models ({unit})' if unit else 'Std. dev. across models',
                    cmap='Purples', add_satellite=add_satellite)
            except Exception as e:
                print(f"⚠️ composite sensitivity-spread grid for {var} failed: {e}")
                traceback.print_exc()
            try:
                make_composite_metric_grid(
                    snr_grids_2d, scenarios, period_order,
                    geom_native, districts_gdf,
                    os.path.join(graphs_dir, f'composite_{var}_sensitivity_snr.png'),
                    title_line1=f'{var_title} \u2014 Sensitivity: Signal-to-Noise Ratio \u2014 {region_name}',
                    colorbar_label='Signal-to-noise ratio (|mean change| / spread)',
                    cmap='YlOrBr', add_satellite=add_satellite)
            except Exception as e:
                print(f"⚠️ composite SNR grid for {var} failed: {e}")
                traceback.print_exc()
    print(f"✅ Model agreement/sensitivity maps saved: {agreement_maps_made}")

    # ── Data-availability summary ────────────────────────────────────────
    # Pinpoints exactly which scenario/period/variable combo had 0 models
    # succeed, which is the #1 reason maps silently don't get made — instead
    # of scrolling back through every ⚠️ line above.
    print("\n📊 Data availability summary:")
    for var in variables:
        print(f"   {var}: historical QDM trained on {len(qdm_cache[var])}/{len(models)} model(s)")
        for scenario in scenarios:
            for start, end, label, tag in future_intervals:
                n = len(corrected_grids[var][scenario][tag])
                flag = '' if n > 0 else '  ⚠️ 0 models succeeded — this is why maps/graphs for this combo are empty; see traceback above'
                print(f"      {scenario}/{tag}: {n}/{len(models)} model(s){flag}")

    print("\n🎉 PIPELINE COMPLETE!")
    return results
