import xarray as xr
import os
from neural_lam import constants as const
import time 
  
     
def copy_era5_subset_reduced():
	print("Opening zarr")
	ds = xr.open_zarr("/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_era5_train001/forcing.zarr", consolidated=True)
	print("Saving to netcdf")

	os.makedirs("/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_era5_train001_netcdf/", exist_ok=True)
	ds.to_netcdf("/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_era5_train001_netcdf/forcing.nc")
	print("Saved")
    
if __name__ == "__main__":
	copy_era5_subset_reduced()
