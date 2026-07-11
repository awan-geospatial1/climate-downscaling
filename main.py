import os, re, itertools, time, traceback, pandas as pd, numpy as np, xarray as xr, geopandas as gpd
from datetime import datetime
from tqdm.auto import tqdm
from shapely.geometry import mapping
from config import _CFG, DEFAULT_NQUANTILES, DEFAULT_QDM_GROUP, DEFAULT_WET_THRESH, DEFAULT_CHUNKS_LATLON
from gee_utils import fetch_reference, fetch_cmip6, regrid_to_reference, clean_time_attrs
from qdm_utils import train_qdm, apply_qdm, adjust_wet_day_frequency
from indices_utils import compute_temperature_indices, compute_precipitation_indices, aggregate_across_models
from plot_utils import plot_fan_chart, make_spatial_map, make_index_panel
from report_utils import write_excel_summary
from logger_utils import setup_logger
from xclim import indices as xci

def _kelvin_to_celsius(da):
    da_c = da - 273.15
    da_c.attrs['units'] = 'degC'
    return da_c

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
    """
    Top-level entry point. Sets up the timestamped run log, executes the
    pipeline, and logs total elapsed time (and the error/traceback on
    failure) before re-raising.
    """
    out_dir = params['output_dir']
    logger, log_path = setup_logger(out_dir)

    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time:%Y-%m-%d %H:%M:%S}")

    try:
        results = _run_pipeline(params, logger)
    except Exception as e:
        end_time = datetime.now()
        logger.error(f"Pipeline FAILED after {end_time - start_time}: {e}")
        logger.error(traceback.format_exc())
        raise

    end_time = datetime.now()
    elapsed = end_time - start_time
    logger.info(f"Pipeline finished successfully at {end_time:%Y-%m-%d %H:%M:%S} (elapsed: {elapsed})")

    return results

def _run_pipeline(params, logger):
    """Core pipeline logic. Every notable step is timestamped in the log file."""
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
        logger.info("GEE already initialised.")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=gee_project_id)
        logger.info(f"GEE initialised with project: {gee_project_id}")
    geom_native, geom_buffered, extent = load_shapefile(shp_path, buffer_km)
    region = ee.Geometry(mapping(geom_buffered))
    logger.info("AOI prepared.")
    os.makedirs(out_dir, exist_ok=True)
    variables = ['tas', 'tasmax', 'tasmin', 'pr']
    for v in variables:
        _CFG[v]['chunks'] = chunks
    # output_dir/
    #   tas/ ssp245/  ssp585/  ensemble/
    #   tasmax/ ...
    #   tasmin/ ...
    #   pr/ ...
    var_dirs = {}
    for v in variables:
        var_dirs[v] = {'root': os.path.join(out_dir, v)}
        for scenario in scenarios:
            scen_dir = os.path.join(out_dir, v, scenario)
            os.makedirs(scen_dir, exist_ok=True)
            var_dirs[v][scenario] = scen_dir
        ens_dir = os.path.join(out_dir, v, 'ensemble')
        os.makedirs(ens_dir, exist_ok=True)
        var_dirs[v]['ensemble'] = ens_dir
    ref_cache = {}
    step_start = time.time()
    for var in tqdm(variables, desc='Reference data (baseline)'):
        ref_cache[var] = fetch_reference(var, b_start, b_end, region, extent, _CFG[var])
        logger.info(f"Fetched reference data for '{var}'.")
    logger.info(f"Reference data step complete in {time.time() - step_start:.1f}s.")
    qdm_cache = {v:{} for v in variables}
    hist_cache = {v:{} for v in variables}
    train_combos = list(itertools.product(variables, models))
    step_start = time.time()
    for var, model in tqdm(train_combos, desc='Training QDM (variable × model)'):
        cfg = _CFG[var]
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
            logger.info(f"QDM trained for {var}/{model}.")
        except Exception as e:
            logger.warning(f"{var}/{model} QDM failed: {e}")
    logger.info(f"QDM training step complete in {time.time() - step_start:.1f}s.")
    corrected_grids = {
        v: {s: {tag: {} for _, _, _, tag in future_intervals} for s in scenarios}
        for v in variables
    }
    proj_combos = [
        (var, scenario, start, end, label, tag, model)
        for var in variables
        for scenario in scenarios
        for start, end, label, tag in future_intervals
        for model in models
        if model in qdm_cache[var]
    ]
    step_start = time.time()
    for var, scenario, start, end, label, tag, model in tqdm(proj_combos, desc='Future projections'):
        cfg = _CFG[var]
        try:
            ds = fetch_cmip6(var, model, scenario, start, end, region, extent, cfg,
                             f'{var}/{model}/{scenario}/{tag}')
            da = regrid_to_reference(ds[var], ref_cache[var]['ref'], cfg['fill_na'])
            corr = apply_qdm(qdm_cache[var][model], da, cfg['clip_min'], cfg['clip_max'], cfg['ref_units'])
            if cfg['wet_adjust']:
                corr = adjust_wet_day_frequency(ref_cache[var]['ref'], corr, thresh=wet_thresh)
            corr = clean_time_attrs(corr)
            out_path = os.path.join(var_dirs[var][scenario], f'qdm_{var}_{model}_{scenario}_{tag}.nc')
            corr.load().to_netcdf(out_path)
            corrected_grids[var][scenario][tag][model] = corr
            logger.info(f"Saved {var}/{model}/{scenario}/{tag} -> {out_path}")
        except Exception as e:
            logger.warning(f"{var}/{model}/{scenario}/{tag} failed: {e}")
    logger.info(f"Future projections step complete in {time.time() - step_start:.1f}s.")
    # --- Ensemble-mean NetCDFs (one per variable/scenario/future-interval) ---
    ens_combos = [
        (var, scenario, tag)
        for var in variables
        for scenario in scenarios
        for _, _, _, tag in future_intervals
    ]
    step_start = time.time()
    for var, scenario, tag in tqdm(ens_combos, desc='Ensemble means'):
        model_das = corrected_grids[var][scenario][tag]
        if not model_das:
            continue
        ens_mean = xr.concat(list(model_das.values()), dim='model').mean(dim='model')
        ens_mean = clean_time_attrs(ens_mean)
        out_path = os.path.join(var_dirs[var]['ensemble'], f'{var}_{scenario}_{tag}_ensemble_mean.nc')
        ens_mean.load().to_netcdf(out_path)
        logger.info(f"Saved ensemble mean -> {out_path}")
    logger.info(f"Ensemble means step complete in {time.time() - step_start:.1f}s.")
    results = {}
    step_start = time.time()
    temp_baseline = compute_temperature_indices(ref_cache['tas']['ref'], ref_cache['tasmax']['ref'],
                                                ref_cache['tasmin']['ref'], b_start, b_end, temp_thresholds)
    precip_baseline = compute_precipitation_indices(ref_cache['pr']['ref'], b_start, b_end,
                                                    precip_thresholds, wet_months, dry_months,
                                                    return_periods, n_boot)
    logger.info(f"Baseline indices computed in {time.time() - step_start:.1f}s.")
    def _wrap_baseline(d):
        # Baseline has no ensemble spread, so mean/p10/p90 are all the same value.
        # gev_return_levels is already a dict of {T: {mean,p10,p90}} and must be
        # left as-is; every other entry (scalar or per-month list) gets wrapped.
        return {k: (v if isinstance(v, dict) else {'mean': v, 'p10': v, 'p90': v})
                for k, v in d.items()}

    results['Baseline'] = {
        'temperature': _wrap_baseline(temp_baseline),
        'precipitation': _wrap_baseline(precip_baseline),
    }
    index_combos = list(itertools.product(scenarios, future_intervals))
    step_start = time.time()
    for scenario, (start, end, label, tag) in tqdm(index_combos, desc='Computing indices'):
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
        logger.info(f"Indices computed for {key}.")
    logger.info(f"Future indices step complete in {time.time() - step_start:.1f}s.")
    rows = []
    for period, groups in results.items():
        for domain, idx_dict in groups.items():
            for idx_name, stats in idx_dict.items():
                if idx_name == 'gev_return_levels':
                    for T, s in stats.items():
                        headline = s.get(headline_stat.get('gev_return_level','p90'), s['mean'])
                        rows.append([period, domain, f'{idx_name}_{T}yr', s['mean'], s['p10'], s['p90'], headline])
                elif isinstance(stats.get('mean'), list):
                    for m in range(12):
                        # Strip a trailing numeric threshold suffix, with or without a unit
                        # letter (e.g. 'su_days_per_month_30C' -> 'su_days_per_month',
                        # 'wetdays_per_month_0.1mm' -> 'wetdays_per_month').
                        base = re.sub(r'_[\d.]+[A-Za-z]*$', '', idx_name)
                        hstat = headline_stat.get(base, 'mean')
                        headline = stats[hstat][m]
                        rows.append([period, domain, f'{idx_name}_m{m+1:02d}', stats['mean'][m], stats['p10'][m], stats['p90'][m], headline])
                else:
                    base = idx_name.split('_mean')[0].split('_p90')[0]
                    hstat = headline_stat.get(base, 'mean')
                    headline = stats.get(hstat, stats['mean'])
                    rows.append([period, domain, idx_name, stats['mean'], stats['p10'], stats['p90'], headline])
    df = pd.DataFrame(rows, columns=['period','domain','index','mean','p10','p90','headline_value'])
    excel_path = os.path.join(out_dir, 'climate_indices_summary.xlsx')
    write_excel_summary(df, excel_path, meta={
        'baseline_period': f'{b_start} to {b_end}',
        'models': models,
        'scenarios': scenarios,
    })
    logger.info(f"Excel summary saved: {excel_path}")
    plot_fan_chart(hist_cache, corrected_grids, 'tas', 'Annual mean temperature (K)',
                   os.path.join(out_dir, 'fanchart_tas.png'), 'Temperature',
                   scenarios, future_intervals, models)
    plot_fan_chart(hist_cache, corrected_grids, 'pr', 'Annual total precipitation (mm)',
                   os.path.join(out_dir, 'fanchart_pr.png'), 'Precipitation',
                   scenarios, future_intervals, models)
    logger.info("Fan charts generated.")
    index_map = {
        'annual_mean_tas': ('tas', 'Mean Annual Temperature', '°C', 'inferno',
                            lambda da: _kelvin_to_celsius(xci.tg_mean(da, freq='YS').mean(dim='time'))),
        'txx': ('tasmax', 'Annual Maximum Temperature (TXx)', '°C', 'inferno',
                lambda da: _kelvin_to_celsius(xci.tx_max(da, freq='YS').mean(dim='time'))),
        'tnn': ('tasmin', 'Annual Minimum Temperature (TNn)', '°C', 'inferno',
                lambda da: _kelvin_to_celsius(xci.tn_min(da, freq='YS').mean(dim='time'))),
        'prcptot': ('pr', 'Annual Total Wet-Day Precipitation (PRCPTOT)', 'mm', 'Blues',
                    lambda da: xci.precip_accumulation(da, freq='YS').mean(dim='time')),
        'rx5day': ('pr', 'Max 5-Day Precipitation (Rx5day)', 'mm', 'Blues',
                   lambda da: xci.max_n_day_precipitation_amount(da, window=5, freq='YS').mean(dim='time')),
        'sdii': ('pr', 'Simple Daily Intensity Index (SDII)', 'mm/wet-day', 'Blues',
                 lambda da: xci.daily_pr_intensity(da, thresh='1 mm/day', freq='YS').mean(dim='time')),
        'cwd': ('pr', 'Consecutive Wet Days (CWD)', 'days', 'Blues',
                lambda da: xci.maximum_consecutive_wet_days(da, thresh='1 mm/day', freq='YS').mean(dim='time')),
    }
    step_start = time.time()
    for idx_name, (var, disp_name, cbar_label, cmap, reducer) in tqdm(index_map.items(), desc='Index panels'):
        panel_grids = {}
        for scenario in scenarios:
            panel_grids[scenario] = {}
            for _, _, _, tag in future_intervals:
                grids = corrected_grids.get(var, {}).get(scenario, {}).get(tag, {})
                if not grids:
                    continue
                per_model = [reducer(da) for da in grids.values()]
                panel_grids[scenario][tag] = xr.concat(per_model, dim='model').mean(dim='model')
        ref_grid = reducer(ref_cache[var]['ref'])
        out_path = os.path.join(var_dirs[var]['ensemble'], f'{idx_name}_panel.png')
        make_index_panel(
            ref_grid, panel_grids, geom_native, out_path,
            title=f'{idx_name.upper()} — {disp_name}',
            subtitle='Reference (baseline) vs. CMIP6 ensemble mean, QDM bias-corrected',
            cbar_label=cbar_label, cmap=cmap,
        )
        logger.info(f"Index panel saved: {out_path}")
    logger.info(f"Index panels step complete in {time.time() - step_start:.1f}s.")
    logger.info("PIPELINE COMPLETE!")
    return results
