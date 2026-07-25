import os, traceback, pandas as pd, numpy as np, xarray as xr, geopandas as gpd
from shapely.geometry import mapping

from config import _CFG, DEFAULT_NQUANTILES, DEFAULT_QDM_GROUP, DEFAULT_WET_THRESH, DEFAULT_CHUNKS_LATLON
from gee_utils import fetch_reference, fetch_cmip6, regrid_to_reference, clean_time_attrs
from qdm_utils import train_qdm, apply_qdm, adjust_wet_day_frequency
from indices_utils import (compute_temperature_indices, compute_precipitation_indices, aggregate_across_models,
                           daily_spatial_series, daily_spatial_ensemble)
from plot_utils import plot_fan_chart, make_spatial_map, plot_index_comparison
from xclim import indices as xci


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
    return geom_native, geom_buffered, extent


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

    headline_stat = params.get('headline_stat', {
        'annual_mean_tas': 'mean', 'annual_mean_tasmax': 'mean',
        'annual_mean_tasmin': 'mean', 'monthly_mean_tas': 'mean',
        'su_days_per_month': 'mean', 'prcptot': 'mean',
        'wet_season_total': 'mean', 'dry_season_total': 'mean',
        'rx1day': 'p90', 'rx5day': 'p90',
        'wetdays_per_month': 'mean', 'gev_return_level': 'p90',
    })

    try:
        ee.Initialize()
        print("✅ GEE already initialised.")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=gee_project_id)
        print(f"✅ GEE initialised with project: {gee_project_id}")

    geom_native, geom_buffered, extent = load_shapefile(shp_path, buffer_km)
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
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(maps_dir, exist_ok=True)
        return data_dir, maps_dir

    def scenario_tables_dir(scenario):
        d = os.path.join(out_dir, scenario, 'tables')
        os.makedirs(d, exist_ok=True)
        return d

    variables = ['tas', 'tasmax', 'tasmin', 'pr']
    for v in variables:
        _CFG[v]['chunks'] = chunks

    ref_cache = {}
    for var in variables:
        ref_cache[var] = fetch_reference(var, b_start, b_end, region, extent, _CFG[var])

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
                        data_dir, _ = scenario_period_dirs(scenario, tag)
                        out_path = os.path.join(data_dir, f'qdm_{var}_{model}.nc')
                        corr.load().to_netcdf(out_path)
                        corrected_grids[var][scenario][tag][model] = corr
                        print(f"💾 {var}/{model}/{scenario}/{tag} saved → {out_path}")
                    except Exception as e:
                        print(f"⚠️ {var}/{model}/{scenario}/{tag} failed: {e}")
                        traceback.print_exc()

    results = {}
    temp_baseline = compute_temperature_indices(ref_cache['tas']['ref'], ref_cache['tasmax']['ref'],
                                                 ref_cache['tasmin']['ref'], b_start, b_end, temp_thresholds)
    precip_baseline = compute_precipitation_indices(ref_cache['pr']['ref'], b_start, b_end,
                                                      precip_thresholds, wet_months, dry_months,
                                                      return_periods, n_boot)
    results['Baseline'] = {
        'temperature': {k: {'mean': v, 'p10': v, 'p90': v} if not isinstance(v, list) else v for k, v in temp_baseline.items()},
        'precipitation': {k: {'mean': v, 'p10': v, 'p90': v} if isinstance(v, (int, float)) else v for k, v in precip_baseline.items()}
    }

    for scenario in scenarios:
        for start, end, label, tag in future_intervals:
            key = f'{scenario}_{tag}'
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
                                                                   wet_months, dry_months, return_periods, n_boot))
            results[key] = {
                'temperature': aggregate_across_models(temp_list, return_periods),
                'precipitation': aggregate_across_models(precip_list, return_periods)
            }

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
    index_map = {
        'annual_mean_tas': ('tas', lambda da: xci.tg_mean(da, freq='YS').mean(dim='time')),
        'prcptot': ('pr', lambda da: xci.precip_accumulation(da, freq='YS').mean(dim='time')),
    }

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
                    _, maps_dir = scenario_period_dirs(scenario, tag)
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
