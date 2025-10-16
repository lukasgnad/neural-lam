import xarray as xr
import xesmf as xe
import cftime
import pandas as pd
import numpy as np
import traceback
import xesmf as xe
import var_dicts as d
import utils as u

# Example: regrid and merge 'q' variable across decades
base_dir = "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049/not_regridded/"
out_dir = "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049_conservative/"
t_chunks = 50

variables = [
    'q','u','v','w','z','t'
    ]

ds = xr.open_dataset(base_dir+f"3D_nextgems_2049_6hourly_0.25deg_atmospherical_selected_levels.nc", chunks={"time": t_chunks})
levels_to_keep = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        
ds = ds.sel(level=levels_to_keep)

# Fix lons and lats
ds['lon'] = ds['lon'] % 360
ds = ds.sortby('lat')
# Initialize output dataset
ds_out = xr.Dataset()

for var in variables:
    variable = var
        
    print(f"Starting variable {variable}")
    try:
        # Reshape if confident about ordering:
        lat_len = len(np.unique(ds['lat'].values))
        lon_len = len(np.unique(ds['lon'].values))
        t_reshaped = ds[variable].data.reshape(ds.sizes['time'], ds.sizes['level'], lat_len, lon_len)

        ds_final = xr.Dataset(
            {variable: (('time', 'level', 'lat', 'lon'), t_reshaped)},
            coords={
                'time': ds['time'],
                'level': ds['level'],
                'lat': np.sort(np.unique(ds['lat'].values)),
                'lon': np.sort(np.unique(ds['lon'].values))
            }
        )

        regridder = u.get_regridder(variable, ds_final)

        # 4. Apply regridding
        if variable == "z":
            print("Applying NaN-ignoring regridding workaround for geopotential")
            data = ds_final[variable]
            mask = xr.where(data.notnull(), 1.0, 0.0)
            data_filled = xr.where(data.notnull(), data, 0.0)

            # Regrid data and mask separately
            data_regridded_sum = regridder(data_filled)
            mask_regridded_sum = regridder(mask)

            # Compute weighted average ignoring NaNs
            ds_regridded_var = data_regridded_sum / mask_regridded_sum

            # Preserve attributes if needed
            ds_regridded_var.attrs = data.attrs

            # Convert to dataset
            ds_regridded = ds_regridded_var.to_dataset(name=variable)
        else:
            ds_regridded = regridder(ds_final[variable])
            ds_regridded = ds_regridded.to_dataset(name=variable)

        # Switch lat and lon to match ERA5 data
        ds_regridded = ds_regridded.transpose('time', 'level', 'lon', 'lat')
        
        # Save half the storage by switching to float32
        ds_regridded = ds_regridded.astype(np.float32)

        ds_out = xr.merge([ds_out, ds_regridded], compat='no_conflicts')
        
    except Exception:
        print(f"Exception occured:")
        print(traceback.format_exc())

print(ds_out)
var_rename_dict_filtered = {
    k: v for k, v in d.var_rename_dict.items()
    if k in ds_out.variables or k in ds_out.dims
}
ds_out = ds_out.rename(var_rename_dict_filtered)


# 6. Write to Zarr (recommended) or NetCDF
ds_out.to_zarr(out_dir + f"3D_nextgems_2049_6hourly_regridded_atmospherical_selected_levels.zarr", mode='w')


print("Saved!")
print("-"*20)
print("")