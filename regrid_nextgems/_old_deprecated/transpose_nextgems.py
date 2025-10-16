import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

base = "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextGEMS_1990_2020_6h-128x64/"

ds = xr.open_dataset(base+"fields.zarr")

# Helper function to reorder dimensions
def reorder_var(var, var_name):
    dims = var.dims
    # Fully dynamic: only reorder if dims present
    new_order = []
    if 'time' in dims:
        new_order.append('time')
    if 'level' in dims:
        new_order.append('level')
    if 'longitude' in dims:
        new_order.append('longitude')
    if 'latitude' in dims:
        new_order.append('latitude')
    # Add any other dims to the end
    for d in dims:
        if d not in new_order:
            new_order.append(d)
    print(f"Reordering variable: {var_name} from {dims} to {new_order}")
    return var.transpose(*new_order)




ATMOSPHERIC_PARAMS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]  # times 13 pressure levels = 78 params

SURFACE_PARAMS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]  # = 5 params
# Total = 83 params
PREPROCESSING_PARAMS = [
    "geopotential_at_surface",
	"land_sea_mask"
]


# --- Atmospheric VARS ---

# Apply reordering to all variables with progress printing
reordered_vars = {}
for var_name in ATMOSPHERIC_PARAMS:
    reordered_vars[var_name] = reorder_var(ds[var_name], var_name)
# Create reordered dataset
ds_reordered = xr.Dataset(reordered_vars, coords=ds.coords)
ds_reordered.attrs = ds.attrs


ds_reordered = ds_reordered.chunk({"time": 16, "level": 13, "longitude": 128, "latitude": 64})

encoding = {var: {"chunks": (16, 13, 128, 64)} for var in ds_reordered.data_vars}

# Write out efficiently
ds_reordered.to_zarr("/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextGEMS_1990_2020_6h-128x64_transposed/fields.zarr", mode="w", encoding=encoding)


# --- Surface VARS ---

# Apply reordering to all variables with progress printing
reordered_vars = {}
for var_name in SURFACE_PARAMS:
    reordered_vars[var_name] = reorder_var(ds[var_name], var_name)
# Create reordered dataset
ds_reordered = xr.Dataset(reordered_vars, coords=ds.coords)
ds_reordered.attrs = ds.attrs


ds_reordered = ds_reordered.chunk({"time": 16, "longitude": 128, "latitude": 64})

encoding = {var: {"chunks": (16, 128, 64)} for var in ds_reordered.data_vars}

# Write out efficiently
ds_reordered.to_zarr("/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextGEMS_1990_2020_6h-128x64_transposed/fields.zarr", mode="a", encoding=encoding)


# --- Timeless VARS ---

# Apply reordering to all variables with progress printing
reordered_vars = {}
for var_name in PREPROCESSING_PARAMS:
    reordered_vars[var_name] = ds[var_name]
# Create reordered dataset
ds_reordered = xr.Dataset(reordered_vars, coords=ds.coords)
ds_reordered.attrs = ds.attrs


ds_reordered = ds_reordered.chunk({"longitude": 128, "latitude": 64})

encoding = {var: {"chunks": (128, 64)} for var in ds_reordered.data_vars}

# Write out efficiently
ds_reordered.to_zarr("/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextGEMS_1990_2020_6h-128x64_transposed/fields.zarr", mode="a", encoding=encoding)



