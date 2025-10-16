import sys
import xarray as xr
import numpy as np
import pandas as pd

base = "/hkfs/work/workspace/scratch/vy6128-nextgems_training_historical/6hourly/"

vars = {#'q': ['3D_nextgems_2010s_6hourly_0.25deg_q.nc'], '3D_nextgems_1990s_6hourly_0.25deg_q_new.nc', '3D_nextgems_2000s_6hourly_0.25deg_q_new.nc', 
        # 't': ['3D_nextgems_2010s_6hourly_0.25deg_t.nc'], '3D_nextgems_1990s_6hourly_0.25deg_t.nc', '3D_nextgems_2000s_6hourly_0.25deg_t_new.nc', 
        #'10u': ['surface_nextgems_1990s_2020_6hourly_0.25deg_surface_uvtmsl.nc'],
        #'10v': ['surface_nextgems_1990s_2020_6hourly_0.25deg_surface_uvtmsl.nc'],
        #'msl': ['surface_nextgems_1990s_2020_6hourly_0.25deg_surface_uvtmsl.nc'],
        '2t': ['surface_nextgems_1990s_2020_6hourly_0.25deg_surface_uvtmsl.nc'],
        # 'u': ['3D_nextgems_2010s_6hourly_0.25deg_u.nc'], #'3D_nextgems_1990s_6hourly_0.25deg_u_new.nc', '3D_nextgems_2000s_6hourly_0.25deg_u_new.nc', 
        # 'v': ['3D_nextgems_2010s_6hourly_0.25deg_v.nc'], # '3D_nextgems_1990s_6hourly_0.25deg_v_new.nc', '3D_nextgems_2000s_6hourly_0.25deg_v.nc', 
        #'w': ['3D_nextgems_1990s_6hourly_0.25deg_w.nc', '3D_nextgems_2000s_6hourly_0.25deg_w_new.nc', '3D_nextgems_2010s_6hourly_0.25deg_w.nc'],
        #'z': ['3D_nextgems_1990s_6hourly_0.25deg_z.nc', '3D_nextgems_2000s_6hourly_0.25deg_z.nc', '3D_nextgems_2010s_6hourly_0.25deg_z.nc'],
        
        #'tp': ['surface_nextgems_1990s_2020_allhours_0.25deg_total_precipitation_new.nc'],# 1994 - ...
        }

for var in vars.keys():
    for filename in vars[var]:
        nan_locations = []
        ds = xr.open_dataset(base + filename, chunks={"time": 50})

        ds['lon'] = ds['lon'] % 360
        ds = ds.sortby('lat')

        # Reshape if confident about ordering:
        lat_len = len(np.unique(ds['lat'].values))
        lon_len = len(np.unique(ds['lon'].values))

        t_reshaped = ds[var].data.reshape(ds.dims['time'], lat_len, lon_len)

        ds_final = xr.Dataset(
            {var: (('time', 'lat', 'lon'), t_reshaped)},
            coords={
                'time': ds['time'],
                'lat': np.sort(np.unique(ds['lat'].values)),
                'lon': np.sort(np.unique(ds['lon'].values))
            }
        )


        fill_value = ds.variables[var]._attrs['missingValue']  # Known missing value indicator in your dataset
        print(f"Processing variable: {var}. Filtering for missing value: {fill_value}")
        for year in range(1990, 2020):
            
            #try:
            print(ds_final['lat'])
            da = ds_final[var].sel(time=str(year)).sel(lat=slice(-20,20))


            print(f"Processing year {year}...")
            # Build missing mask: True where either NaN or 9999
            missing_mask = (da < 250)

            if missing_mask.any():
                # Get indices where missing
                idxs = np.argwhere(missing_mask.values)

                # Collect coordinates for each missing point
                for idx in idxs:
                    coords = {}
                    isel_dict = {}
                    for axis, dim in enumerate(da.dims):
                        coord_val = da[dim].values[idx[axis]]
                        coords[dim] = idx[axis]
                        isel_dict[dim] = idx[axis]
                    coords["year"] = year
                    coords["value"] = da.isel(isel_dict).values.item()
                    nan_locations.append(coords)

                df_nan_locations = pd.DataFrame(nan_locations)
                print(df_nan_locations)
                df_nan_locations.to_csv(f"missing_csv/t2m_{filename}_{year}.csv", index=False)
            # except Exception as e:
            #     print(e)
            #     continue

        if len(nan_locations) > 0:
            # Convert to DataFrame for inspection or export
            df_nan_locations = pd.DataFrame(nan_locations)
            print(df_nan_locations)

            # Optional: Save for later processing
        
            df_nan_locations.to_csv(f"missing_csv/t2m_{filename}.csv", index=False)
        else:
            print("Nothing found!")
        print("-"*30)
        print()
