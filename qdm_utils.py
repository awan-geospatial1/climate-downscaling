import xsdba as sdba
import xarray as xr, numpy as np

def train_qdm(ref_da, hist_da, kind, nquantiles=50, group='time.month'):
    return sdba.QuantileDeltaMapping.train(
        ref=ref_da.chunk({'time': -1}), hist=hist_da.chunk({'time': -1}),
        nquantiles=nquantiles, group=group, kind=kind,
    )

def apply_qdm(qdm_obj, fut_da, clip_min, clip_max, units):
    corrected = qdm_obj.adjust(fut_da.chunk({'time': -1}))
    corrected = corrected.where(np.isfinite(corrected), np.nan).clip(min=clip_min, max=clip_max)
    corrected.attrs['units'] = units
    return corrected

def adjust_wet_day_frequency(ref_da, sim_da, thresh=0.1):
    ref_da = ref_da.where(np.isfinite(ref_da), np.nan).transpose('time', ...)
    sim_da = sim_da.where(np.isfinite(sim_da), np.nan).transpose('time', ...)
    adjusted = sim_da.copy(deep=True)
    for month, ref_month in ref_da.groupby('time.month'):
        sim_month = sim_da.sel(time=sim_da['time.month'] == month)
        if len(sim_month.time) == 0: continue
        n_sim_days = len(sim_month.time)
        ref_wet_frac = (ref_month >= thresh).mean(dim='time', skipna=True)
        target_wet_days = np.round(ref_wet_frac * n_sim_days).astype(int)
        actual_wet_days = (sim_month >= thresh).sum(dim='time', skipna=True)
        n_remove = (actual_wet_days - target_wet_days).clip(min=0)
        n_remove = xr.where(np.isfinite(ref_wet_frac), n_remove, 0)
        def _zero_smallest_wet(pixel_values, n_to_remove):
            arr = np.array(pixel_values, copy=True)
            n = int(n_to_remove)
            if n <= 0 or np.all(np.isnan(arr)): return arr
            wet_idx = np.where((arr >= thresh) & np.isfinite(arr))[0]
            if len(wet_idx) == 0: return arr
            sorted_wet = wet_idx[np.argsort(arr[wet_idx])]
            arr[sorted_wet[:n]] = 0.0
            return arr
        adjusted_month = xr.apply_ufunc(
            _zero_smallest_wet, sim_month, n_remove,
            input_core_dims=[['time'], []], output_core_dims=[['time']],
            vectorize=True, dask='parallelized', output_dtypes=[sim_month.dtype],
            dask_gufunc_kwargs={'output_sizes': {'time': n_sim_days}},
        )
        adjusted.loc[dict(time=sim_month.time)] = adjusted_month
    return adjusted
