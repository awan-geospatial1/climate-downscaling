import numpy as np, xarray as xr
from xclim import indices as xci
from scipy.stats import genextreme

def spatial_mean(da):
    dims = [d for d in ('lat','lon') if d in da.dims]
    return da.mean(dim=dims, skipna=True) if dims else da

def gev_return_levels(annual_max_series, return_periods, n_boot=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    x = np.asarray(annual_max_series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {T: {'mean':np.nan, 'p10':np.nan, 'p90':np.nan} for T in return_periods}
    boot = {T: [] for T in return_periods}
    n = len(x)
    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        try:
            shape, loc, scale = genextreme.fit(sample)
        except Exception:
            continue
        for T in return_periods:
            boot[T].append(genextreme.ppf(1 - 1.0/T, shape, loc=loc, scale=scale))
    out = {}
    for T in return_periods:
        vals = np.array(boot[T]); vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[T] = {'mean':np.nan, 'p10':np.nan, 'p90':np.nan}
        else:
            out[T] = {'mean':float(np.mean(vals)), 'p10':float(np.percentile(vals,10)),
                      'p90':float(np.percentile(vals,90))}
    return out

def compute_temperature_indices(tas_da, tasmax_da, tasmin_da, start, end, temp_thresholds):
    tas_1d = spatial_mean(tas_da).sel(time=slice(start, end))
    tasmax_1d = spatial_mean(tasmax_da).sel(time=slice(start, end))
    tasmin_1d = spatial_mean(tasmin_da).sel(time=slice(start, end))
    tas_1d.attrs['units'] = tasmax_1d.attrs['units'] = tasmin_1d.attrs['units'] = 'K'
    out = {}
    out['annual_mean_tas'] = float(xci.tg_mean(tas_1d, freq='YS').mean())
    out['annual_mean_tasmax'] = float(xci.tx_mean(tasmax_1d, freq='YS').mean())
    out['annual_mean_tasmin'] = float(xci.tn_mean(tasmin_1d, freq='YS').mean())
    monthly = xci.tg_mean(tas_1d, freq='MS')
    out['monthly_mean_tas'] = monthly.groupby('time.month').mean().values.tolist()
    for thr in temp_thresholds:
        cnt = xci.tx_days_above(tasmax_1d, thresh=f'{thr} degC', freq='MS')
        out[f'su_days_per_month_{thr:g}C'] = cnt.groupby('time.month').mean().values.tolist()
    return out

def compute_precipitation_indices(pr_da, start, end, precip_thresholds, wet_months, dry_months, return_periods, n_boot):
    pr_1d = spatial_mean(pr_da).sel(time=slice(start, end))
    pr_1d.attrs['units'] = 'mm/d'
    out = {}
    annual_total = xci.precip_accumulation(pr_1d, freq='YS')
    out['prcptot'] = float(annual_total.mean())
    wet = pr_1d.sel(time=pr_1d['time.month'].isin(wet_months))
    dry = pr_1d.sel(time=pr_1d['time.month'].isin(dry_months))
    out['wet_season_total'] = float(xci.precip_accumulation(wet, freq='YS').mean())
    out['dry_season_total'] = float(xci.precip_accumulation(dry, freq='YS').mean())
    for thr in precip_thresholds:
        cnt = xci.wetdays(pr_1d, thresh=f'{thr} mm/day', freq='MS')
        out[f'wetdays_per_month_{thr:g}mm'] = cnt.groupby('time.month').mean().values.tolist()
    rx1_annual = xci.max_1day_precipitation_amount(pr_1d, freq='YS')
    out['rx1day_mean'] = float(rx1_annual.mean())
    out['rx1day_p90'] = float(rx1_annual.quantile(0.90))
    out['gev_return_levels'] = gev_return_levels(rx1_annual.values, return_periods, n_boot)
    return out

def aggregate_across_models(list_of_index_dicts, return_periods):
    if not list_of_index_dicts: return {}
    out = {}
    keys = list_of_index_dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in list_of_index_dicts if k in d]
        if k == 'gev_return_levels':
            merged = {}
            for T in return_periods:
                m = np.nanmean([v[T]['mean'] for v in vals])
                p10 = np.nanmean([v[T]['p10'] for v in vals])
                p90 = np.nanmean([v[T]['p90'] for v in vals])
                merged[T] = {'mean':float(m), 'p10':float(p10), 'p90':float(p90)}
            out[k] = merged
        elif isinstance(vals[0], list):
            arr = np.array(vals)
            out[k] = {'mean':arr.mean(axis=0).tolist(),
                      'p10':np.percentile(arr,10,axis=0).tolist(),
                      'p90':np.percentile(arr,90,axis=0).tolist()}
        else:
            arr = np.array(vals, dtype=float)
            out[k] = {'mean':float(np.nanmean(arr)),
                      'p10':float(np.nanpercentile(arr,10)),
                      'p90':float(np.nanpercentile(arr,90))}
    return out
