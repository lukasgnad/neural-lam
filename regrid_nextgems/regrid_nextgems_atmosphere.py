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
base_dir = "/hkfs/work/workspace/scratch/vy6128-nextgems_training_historical/6hourly/"
out_dir = "/hkfs/work/workspace/scratch/xo8179-nextgems_regridded/try2/conservative/"
t_chunks = 50


    
years = [1990, 2000, 2010]
variables = [
    'q','u','v','w','z','t'
    ]

datasets = [(2000, 'z'),(2010, 'q'),(2010, 'u'),(2010, 'v'),(2010, 'w'),(2010, 'z'), #'(2010, 't'),(1990, 'w'),(1990, 'z'),(1990, 't'),(2000, 'v'),'
          (1990, 'q_new'),(1990, 'u_new'),(1990, 'v_new'),(2000, 'q_new'),(2000, 'w_new'),(2000, 't_new'),(2000, 'u_new')]

for (year, var) in datasets:
    variable = var[:-4] if var.endswith('_new') else var
    # Somehow, the 'time" dimension is corrupted sometimes for other datasets, but never for temperature -> Use this for all datasets
    if (year == 2000):
        ds_time = xr.open_dataset(base_dir+f"3D_nextgems_{year}s_6hourly_0.25deg_t_new.nc", chunks={"time": t_chunks})['time']
    else:
        ds_time = xr.open_dataset(base_dir+f"3D_nextgems_{year}s_6hourly_0.25deg_t.nc", chunks={"time": t_chunks})['time']
        
    print(f"Starting year {year} - variable {variable}")
    try:
        levels_to_keep = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

        if not variable == "t":
            ds = xr.open_dataset(base_dir+f"3D_nextgems_{year}s_6hourly_0.25deg_{var}.nc", decode_times=False, chunks={"time": t_chunks})
            ds['time'] = ds_time
        else:
            ds = xr.open_dataset(base_dir+f"3D_nextgems_{year}s_6hourly_0.25deg_{var}.nc", chunks={"time": t_chunks})
        
        ds = ds.sel(level=levels_to_keep)

        # Fix lons and lats
        ds['lon'] = ds['lon'] % 360
        ds = ds.sortby('lat')

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
        
        var_rename_dict_filtered = {
            k: v for k, v in d.var_rename_dict.items()
            if k in ds_regridded.variables or k in ds_regridded.dims
        }
        ds_regridded = ds_regridded.rename(var_rename_dict_filtered)
        
        
        # 6. Write to Zarr (recommended) or NetCDF
        ds_regridded.to_zarr(out_dir + f"3D_nextgems_{year}s_6hourly_128x64_{variable}_no_nans.zarr", mode='w')
        
        del ds
        del ds_final
        del ds_regridded
        
        print("Saved!")
        print("-"*20)
        print("")
    except Exception:
        print(f"Exception occured:")
        print(traceback.format_exc())
        print(f"Skipping dataset {year}-{variable}...")
    del ds_time