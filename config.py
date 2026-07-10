# config.py - static configurations
VARIABLES = ['tas', 'tasmax', 'tasmin', 'pr']
_CFG = {
    'pr': dict(
        ref_collection='UCSB-CHG/CHIRPS/DAILY',
        ref_band='precipitation',
        ref_units='mm/d',
        ref_name='CHIRPS',
        qdm_kind='*',
        wet_adjust=True,
        cmip6_unit_factor=86400.0,
        clip_min=0.0, clip_max=1000.0, fill_na=0.0, scale=5566,
    ),
    'tas': dict(
        ref_collection='ECMWF/ERA5_LAND/DAILY_AGGR',
        ref_band='temperature_2m',
        ref_units='K',
        ref_name='ERA5-Land',
        qdm_kind='+',
        wet_adjust=False,
        cmip6_unit_factor=1.0,
        clip_min=200.0, clip_max=350.0, fill_na=280.0, scale=11132,
    ),
    'tasmax': dict(
        ref_collection='ECMWF/ERA5_LAND/DAILY_AGGR',
        ref_band='temperature_2m_max',
        ref_units='K',
        ref_name='ERA5-Land',
        qdm_kind='+',
        wet_adjust=False,
        cmip6_unit_factor=1.0,
        clip_min=200.0, clip_max=350.0, fill_na=290.0, scale=11132,
    ),
    'tasmin': dict(
        ref_collection='ECMWF/ERA5_LAND/DAILY_AGGR',
        ref_band='temperature_2m_min',
        ref_units='K',
        ref_name='ERA5-Land',
        qdm_kind='+',
        wet_adjust=False,
        cmip6_unit_factor=1.0,
        clip_min=200.0, clip_max=350.0, fill_na=275.0, scale=11132,
    ),
}
DEFAULT_NQUANTILES = 50
DEFAULT_QDM_GROUP = 'time.month'
DEFAULT_WET_THRESH = 0.1
DEFAULT_CHUNKS_LATLON = {'lat': 50, 'lon': 50}
