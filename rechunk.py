target_chunks = {
    "specific_humidity": {"time": 1, "longitude": 128, "latitude": 64, "level": 6},
    "temperature": {"time": 1, "longitude": 128, "latitude": 64, "level": 6},
    "u_component_of_wind": {"time": 1, "longitude": 128, "latitude": 64, "level": 6},
    "v_component_of_wind": {"time": 1, "longitude": 128, "latitude": 64, "level": 6},
    "vertical_velocity": {"time": 1, "longitude": 128, "latitude": 64, "level": 6},
    "relative_vorticity": {"time": 1, "longitude": 128, "latitude": 64, "level": 6},
    "mean_sea_level_pressure": {"time": 1, "longitude": 128, "latitude": 64},
    "geopotential_500": {"time": 1, "longitude": 128, "latitude": 64},
    "total_precipitation_6hr": {"time": 1, "longitude": 128, "latitude": 64},
    "surface_temperature": {"time": 1, "longitude": 128, "latitude": 64},
    "surface_u_component_of_wind": {"time": 1, "longitude": 128, "latitude": 64},
    "surface_v_component_of_wind": {"time": 1, "longitude": 128, "latitude": 64},
    "geopotential_at_surface": {"longitude": 128, "latitude": 64},
	"land_sea_mask": {"longitude": 128, "latitude": 64},
    "time": None,  # don't rechunk this array
    "longitude": None,
    "latitude": None,
    "level": None,
}
import xarray as xr
import rechunker

source = xr.open_zarr("/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_long_chunks/fields.zarr")
target_store = "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative/fields_new.zarr"
temp_store = "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative/temp.zarr"

import shutil

shutil.rmtree(target_store, ignore_errors=True)
shutil.rmtree(temp_store, ignore_errors=True)


# Plan rechunking
plan = rechunker.rechunk(
    source, 
    target_chunks=target_chunks, 
    target_store=target_store,
    temp_store=temp_store,
    max_mem="200GB"
)

# Execute
from dask.diagnostics import ProgressBar

with ProgressBar():
    plan.execute()