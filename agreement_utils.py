"""
agreement_utils.py - Model agreement and sensitivity (spread / signal-to-
noise) analysis, per variable / scenario / period, reusing data the
pipeline already has loaded (`corrected_grids[var][scenario][tag]`, a
dict of model -> daily xr.DataArray) instead of re-reading NetCDFs from
disk. Mirrors the same statistics as the standalone AJK agreement/
sensitivity script, adapted to this repo's actual data flow and folder
layout, and reusing plot_utils.make_spatial_map for rendering instead of
a separate multi-panel figure builder, to stay consistent with how every
other spatial map in this pipeline is drawn.

MODEL AGREEMENT: at every grid cell, the % of models whose own change (vs
the baseline reference) has the same sign as the ensemble-mean change.

SENSITIVITY (two maps):
  - Spread: standard deviation of the models' change values, in the
    variable's own units (temperature: degC: precipitation: % points).
  - Signal-to-noise ratio (SNR): |ensemble-mean change| / spread.

See the standalone AJK script's docstring (from the earlier conversation)
for the full explanation of what these numbers mean and why they're
reported as a pair.
"""
import numpy as np
import xarray as xr


# Cells below this baseline (mm/year-equivalent, but applied at the daily-
# annual-climatology stage below) are excluded from percent-change stats,
# for the same reason as in the AJK script: dividing by a near-zero
# baseline produces mathematically unstable, meaningless percentages, not
# genuine large changes. Adjust to match your AOI's climate if most of it
# is very dry (lower this) or you're seeing this exclude too much (raise
# with caution -- it exists specifically to prevent runaway %-change
# values, not to hide real signal).
PR_MIN_BASELINE_MM = 30.0


def _annual_climatology(da, agg):
    """(lat, lon) climatology: mean-of-annual-means (agg='mean', for
    temperature) or mean-of-annual-totals (agg='sum', for precipitation)."""
    annual = da.resample(time='YS').sum() if agg == 'sum' else da.resample(time='YS').mean()
    return annual.mean(dim='time')


def per_model_change_stack(var, baseline_da, grids_by_model, start, end, change_mode):
    """
    baseline_da        : single reference xr.DataArray (time, lat, lon) --
                          e.g. ref_cache[var]['ref'].
    grids_by_model      : dict[model] -> bias-corrected xr.DataArray
                          (time, lat, lon) for one scenario/period, e.g.
                          corrected_grids[var][scenario][tag].
    start, end          : the future period's date bounds (same values
                          passed to compute_*_indices elsewhere), used to
                          slice each model's series before taking its
                          climatology.
    change_mode         : 'absolute' (temperature) or 'percent' (precip).

    Returns (stack, used_models, base_clim):
      stack       : np.ndarray (n_models, lat, lon) of per-model change
                    values, or None if fewer than 2 models were usable.
      used_models  : list of model names actually included, in stack order.
      base_clim    : the baseline (lat, lon) xr.DataArray climatology --
                    kept around so callers can reuse its coords to wrap
                    result arrays (agreement/spread/SNR) back into
                    DataArrays without recomputing anything.
    """
    agg = 'sum' if change_mode == 'percent' else 'mean'
    base_clim = _annual_climatology(baseline_da, agg)

    changes, used = [], []
    for model, da in grids_by_model.items():
        try:
            fut_clim = _annual_climatology(da.sel(time=slice(start, end)), agg)
            fut_on_base = fut_clim.interp(lat=base_clim.lat, lon=base_clim.lon, method='linear')
            fut_on_base = fut_on_base.fillna(
                fut_clim.interp(lat=base_clim.lat, lon=base_clim.lon, method='nearest'))

            if change_mode == 'percent':
                denom = base_clim.where(np.abs(base_clim) >= PR_MIN_BASELINE_MM)
                with np.errstate(divide='ignore', invalid='ignore'):
                    diff = ((fut_on_base - base_clim) / denom) * 100.0
            else:
                # Both grids are Kelvin at this point -- a DELTA in Kelvin
                # equals the same delta in degC, so no conversion needed.
                diff = fut_on_base - base_clim

            changes.append(diff.values)
            used.append(model)
        except Exception as e:
            print(f"    [skip] {var} {model}: {e}")

    if len(changes) < 2:
        return None, used, base_clim
    return np.stack(changes, axis=0), used, base_clim


def compute_agreement_and_sensitivity(stack):
    """stack: (n_models, lat, lon) of per-model change values.
    Returns (agreement_pct, spread, snr), each (lat, lon)."""
    ens_mean = np.nanmean(stack, axis=0)
    ens_sign = np.sign(ens_mean)

    model_signs = np.sign(stack)
    with np.errstate(invalid='ignore'):
        agree_mask = (model_signs == ens_sign[np.newaxis, :, :])
    agreement_pct = np.mean(agree_mask, axis=0) * 100
    agreement_pct = np.where(ens_sign == 0, np.nan, agreement_pct)

    spread = np.nanstd(stack, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.abs(ens_mean) / spread
    snr = np.where(spread < 1e-6, np.nan, snr)

    return agreement_pct, spread, snr


def make_agreement_sensitivity_maps(var, cfg, scenario, tag, start, end, baseline_da,
                                     grids_by_model, geom_native, maps_dir,
                                     make_spatial_map_fn, add_satellite=False):
    """
    Computes agreement/spread/SNR for one variable/scenario/period and
    renders each as its own map via the pipeline's existing
    `make_spatial_map` (passed in as `make_spatial_map_fn` to avoid a
    circular import with plot_utils), saved into the same `maps_dir` every
    other spatial map for this scenario/period already goes into.

    Returns the number of maps successfully saved (0-3), so callers can
    total it up the same way main.py already tracks `maps_made`.
    """
    change_mode = 'percent' if var == 'pr' else 'absolute'
    stack, used_models, base_clim = per_model_change_stack(
        var, baseline_da, grids_by_model, start, end, change_mode)

    if stack is None:
        print(f"    [ERROR] Only {len(used_models)} usable model(s) for "
              f"{var}/{scenario}/{tag} agreement/sensitivity; need >= 2, skipping.")
        return 0

    agreement, spread, snr = compute_agreement_and_sensitivity(stack)
    saved = 0

    specs = [
        ('model_agreement', agreement, 'RdYlGn', '% of models agreeing on direction of change'),
        ('sensitivity_spread', spread, 'Purples', 'Inter-model spread (std. dev. of change)'),
        ('sensitivity_snr', snr, 'YlOrBr', 'Signal-to-noise ratio (|mean change| / spread)'),
    ]
    for suffix, values, cmap, subtitle in specs:
        try:
            da_2d = base_clim.copy(data=values)
            out_path = f'{maps_dir}/{var}_{suffix}_{scenario}_{tag}.png'
            make_spatial_map_fn(da_2d, geom_native, out_path,
                                 title=f'{var} {subtitle} \u2014 {scenario.upper()} ({tag})',
                                 cmap=cmap, add_satellite=add_satellite)
            saved += 1
        except Exception as e:
            print(f"    [ERROR] {var}/{scenario}/{tag} {suffix} map failed: {e}")

    print(f"    used {len(used_models)} model(s) for agreement/sensitivity: {used_models}")
    return saved
