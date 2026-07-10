import ee, xarray as xr, pandas as pd, numpy as np
from tqdm.auto import tqdm
from config import _CFG

def clean_time_attrs(obj):
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        if 'time' in obj.coords and 'calendar' in obj['time'].attrs:
            del obj['time'].attrs['calendar']
    return obj

def clean_values(da, clip_min, clip_max):
    da = da.where(np.isfinite(da), np.nan)
    return da.clip(min=clip_min, max=clip_max)

def regrid_to_reference(src_da, ref_da, fill_na):
    out = src_da.interp(lat=ref_da.lat, lon=ref_da.lon, method='linear')
    out = out.where(np.isfinite(out), np.nan).fillna(fill_na)
    out.attrs['units'] = src_da.attrs.get('units', ref_da.attrs.get('units'))
    return out

def _make_xee_projection(extent, scale):
    lon_min, lat_min, lon_max, lat_max = extent
    deg = scale / 111_320.0
    n_lon = int(np.ceil((lon_max - lon_min) / deg)) + 1
    n_lat = int(np.ceil((lat_max - lat_min) / deg)) + 1
    crs_transform = (deg, 0, lon_min, 0, -deg, lat_max)
    return dict(crs='EPSG:4326', crs_transform=crs_transform, shape_2d=(n_lat, n_lon))

def _xee_open(ic_chunk, proj):
    ds = xr.open_dataset(ic_chunk, engine='ee', crs=proj['crs'],
                         crs_transform=proj['crs_transform'], shape_2d=proj['shape_2d'])
    rename = {}
    if 'x' in ds.dims and 'lon' not in ds.dims: rename['x'] = 'lon'
    if 'y' in ds.dims and 'lat' not in ds.dims: rename['y'] = 'lat'
    return ds.rename(rename) if rename else ds

def _date_chunks(start, end, years=5):
    chunks, current = [], start
    while current <= end:
        chunk_end = min(current + pd.DateOffset(years=years) - pd.DateOffset(days=1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + pd.DateOffset(days=1)
    return chunks

def _fetch_chunked(collection, start_date, end_date, proj, label=''):
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    chunks = []
    date_chunks = _date_chunks(start, end)
    for current, chunk_end in tqdm(date_chunks, desc=f'      ↳ {label}', unit='chunk', leave=False):
        filter_end = chunk_end + pd.DateOffset(days=1)
        ic_chunk = collection.filterDate(current.strftime('%Y-%m-%d'), filter_end.strftime('%Y-%m-%d'))
        chunks.append(_xee_open(ic_chunk, proj))
    return xr.concat(chunks, dim='time').sortby('time')

def fetch_reference(var, start_date, end_date, region, extent, cfg):
    proj = _make_xee_projection(extent, cfg['scale'])
    tqdm.write(f"\n── Fetching reference for {var} ({cfg['ref_name']}) ─────────────")
    ic = ee.ImageCollection(cfg['ref_collection']).filterBounds(region).select(cfg['ref_band'])
    ds = _fetch_chunked(ic, start_date, end_date, proj, label=cfg['ref_name'])
    ds = ds.rename({cfg['ref_band']: 'ref'})
    ds['ref'].attrs['units'] = cfg['ref_units']
    ds = ds.convert_calendar('standard')
    ds['ref'] = clean_values(ds['ref'], cfg['clip_min'], cfg['clip_max'])
    return clean_time_attrs(ds).chunk({'time': -1, **cfg.get('chunks', {'lat':50,'lon':50})})

def fetch_cmip6(var, model, scenario, start_date, end_date, region, extent, cfg, label=''):
    proj = _make_xee_projection(extent, cfg['scale'])
    tqdm.write(f"   ── Fetching CMIP6 {var}: {label} ──")
    ic = (ee.ImageCollection('NASA/GDDP-CMIP6')
          .filterBounds(region)
          .filter(ee.Filter.eq('model', model))
          .filter(ee.Filter.eq('scenario', scenario))
          .select(var))
    ds = _fetch_chunked(ic, start_date, end_date, proj, label=label)
    if cfg['cmip6_unit_factor'] != 1.0:
        ds[var] = ds[var] * cfg['cmip6_unit_factor']
    ds[var].attrs['units'] = cfg['ref_units']
    ds = ds.convert_calendar('standard')
    ds[var] = clean_values(ds[var], cfg['clip_min'], cfg['clip_max'])
    return clean_time_attrs(ds).chunk({'time': -1, **cfg.get('chunks', {'lat':50,'lon':50})})
