import xarray as xr

def main():
    # Path to your NetCDF file (adjust accordingly)
    #nc_path = "/hkfs/work/workspace/scratch/vy6128-nextgems_training_historical/surface_nextgems_1990s_2020_1200_0.25deg_surface_uvtmsl.nc"
    nc_path = "/hkfs/work/workspace/scratch/vy6128-nextgems_training_historical/3D_nextgems_1990s_1200_0.25deg_z.nc"
    
    # Open the NetCDF file using xarray
    ds = xr.open_dataset(nc_path)
    
    # Print dataset metadata
    print("Available variables:", list(ds.data_vars))
    print("Available coordinates:", list(ds.coords))
    
    # Show time range to verify
    print("Time range:", ds.time.values[0], "to", ds.time.values[-1])
    
    # Select a specific day (e.g., 2020-07-01)
    selected_day = ds.sel(time='2020-07-01', method='nearest')  # or use .sel(time=slice(...)) for a range
    
    # Print variable values for selected day
    print("\nExample data for 2020-07-01:")
    for var in list(ds.data_vars)[:3]:  # print first 3 variables as an example
        # Print detailed variable information
        print(f"--- Variable: {var} ---")
        print(f"Dimensions: {selected_day[var].dims}")
        print(f"Shape: {selected_day[var].shape}")
        print(f"Dtype: {selected_day[var].dtype}")
        print(f"Attributes: {selected_day[var].attrs}")
        print(f"Coordinates: {list(selected_day[var].coords)}")
        print(f"Sample data:\n{selected_day[var].values}")
        
        print(f"{var}:")
        print(selected_day[var].values)
        print("-" * 40)

if __name__ == "__main__":
    main()
