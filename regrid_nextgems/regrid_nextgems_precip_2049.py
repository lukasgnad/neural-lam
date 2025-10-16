import xarray as xr
import xesmf as xe
import cftime
import pandas as pd
import numpy as np
import traceback
import dask
import xesmf as xe
import var_dicts as d
import utils as u

# Example: regrid and merge 'q' variable across decades
base_dir = "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049/not_regridded/"
out_dir = "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049_conservative/"
t_chunks = 50

ds = xr.open_dataset(base_dir+f"2D_nextgems_2049_6hourly_0.25deg_tp.nc", chunks={"time": t_chunks})

# Weirdly, Precipiation contains one timestep more than all other nextgems datasets -> Drop the last timestep
ds = ds.isel(time=slice(0,-1))

ds['lon'] = ds['lon'] % 360
ds = ds.sortby('lat')

# Reshape if confident about ordering:
lat_len = len(np.unique(ds['lat'].values))
lon_len = len(np.unique(ds['lon'].values))
time_len = ds.sizes['time']

for variable in ['tp']:
    #for variable in variables:
    try:

        t_reshaped = ds[variable].data.reshape(ds.sizes['time'], lat_len, lon_len)

        ds_final = xr.Dataset(
            {variable: (('time', 'lat', 'lon'), t_reshaped)},
            coords={
                'time': ds['time'],
                'lat': np.sort(np.unique(ds['lat'].values)),
                'lon': np.sort(np.unique(ds['lon'].values))
            }
        )

        regridder = u.get_regridder(variable, ds_final)

        # 4. Apply regridding
        ds_regridded = regridder(ds_final[variable])
        ds_regridded = ds_regridded.to_dataset(name=variable)

        # Switch lat and lon to match ERA5 data
        ds_regridded = ds_regridded.transpose('time', 'lon', 'lat')
        
        # Save half the storage by switching to float32
        ds_regridded = ds_regridded.astype(np.float32)
        
        var_rename_dict_filtered = {
            k: v for k, v in d.var_rename_dict.items()
            if k in ds_regridded.variables or k in ds_regridded.dims
        }
        ds_regridded = ds_regridded.rename(var_rename_dict_filtered)
        
        # 6. Write to Zarr (recommended) or NetCDF
        ds_regridded.to_zarr(out_dir + f"2D_nextgems_2049_6hourly_regridded_tp.zarr", mode='w')
        
        del ds_final
        del ds_regridded
        
        print("Saved!")
        print("-"*20)
        print("")
    except Exception:
        print(f"Exception occured:")
        print(traceback.format_exc())
        print(f"Skipping dataset {variable}...")