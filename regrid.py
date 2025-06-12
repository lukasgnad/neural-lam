from argparse import ArgumentParser
import xarray as xr
import xesmf as xe
import numpy as np
import neural_lam.constants as c
import os

def start_regridding(input_zarr:str, out_zarr:str, out_zarr_short:str):
    # Open original dataset
    ds = xr.open_zarr(input_zarr, consolidated=True)

    ds = ds.rename({
        "latitude": "lat",
        "longitude": "lon"
    })


    ds_chunked = ds
    # Chunk input data by time (adjust chunk size as needed)
    # ds_chunked = ds.chunk({"time": 50})

    # Define 3° grid
    lon_coarse = np.arange(-180 + 1.5, 180, 3)
    lat_coarse = np.arange(-90 + 1.5, 90, 3)

    ds_out_grid = xr.Dataset(
        {
            "lon": (["lon"], lon_coarse),
            "lat": (["lat"], lat_coarse),
        }
    )

    # Create regridders for methods you need
    regridder_bilinear = xe.Regridder(ds, ds_out_grid, "bilinear", periodic=True)
    regridder_conservative = xe.Regridder(ds, ds_out_grid, "conservative", periodic=True)
    regridder_nearest = xe.Regridder(ds, ds_out_grid, "nearest_s2d", periodic=True)

    # Define variable groups (based on your earlier question)
    bilinear_vars = [
        "geopotential", "temperature", "vertical_velocity",
        "2m_temperature", "mean_sea_level_pressure", "geopotential_at_surface",
        "u_component_of_wind", "v_component_of_wind",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
    ]
    conservative_vars = ["specific_humidity", "total_precipitation_6hr"]
    nearest_vars = ["land_sea_mask"]

    # Make sure output dir is clean
    if os.path.exists(out_zarr):
        import shutil
        shutil.rmtree(out_zarr)
        
    if os.path.exists(out_zarr_short):
        import shutil
        shutil.rmtree(out_zarr_short)

    START_DATE = "1980-01-01"
    END_DATE = "2022-12-31"

    # Loop over variables and regrid + save incrementally
    for var in c.ATMOSPHERIC_PARAMS + c.SURFACE_PARAMS + c.PREPROCESSING_PARAMS:
        print(f"Processing variable: {var}")
        da = ds_chunked[var]

        if var in bilinear_vars:
            da_regrid = regridder_bilinear(da)
        elif var in conservative_vars:
            da_regrid = regridder_conservative(da)
        elif var in nearest_vars:
            da_regrid = regridder_nearest(da)
        else:
            print(f"Warning: {var} not assigned a regridding method, skipping.")
            continue


        # Append to Zarr store
        #da_regrid.to_dataset(name=var).to_zarr(output_zarr, mode="a", consolidated=True)
        
        da_out = da_regrid.to_dataset(name=var)
        da_out = da_out.rename({
            "lat": "latitude",
            "lon": "longitude"
        })
        
        
        dims = list(da_out.dims)
        # If level is present
        if "level" in dims:
            new_order = ("time", "longitude", "latitude", "level")
        else:
            if "time" in dims:
                new_order = ("time", "longitude", "latitude")
            else:
                new_order = ("longitude", "latitude")
        
        da_out = da_out.transpose(*new_order)
        
        
        
        da_out.to_zarr(out_zarr, mode="a", consolidated=True)

        if "time" in dims:
            ds = da_out.sel(time=slice(START_DATE, END_DATE))
        else:
            ds = da_out

        print(f"Saving to local path: {out_zarr_short}")
        ds.to_zarr(out_zarr_short, consolidated=True, mode="a")

if __name__ == "__main__":
	"""
	Pre-compute parameter weights to be used in loss function
	"""
	parser = ArgumentParser(description="Training arguments")
	parser.add_argument(
		"--input",
		type=str,
		default="/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_era5_1980_2022/fields.zarr",
		help="Input zarr",
	)
	parser.add_argument(
		"--out",
		type=str,
		default="/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_era5_1980_2022_3deg/fields.zarr",
		help="Output zarr",
	)
	parser.add_argument(
		"--out_short",
		type=str,
		default="/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_era5_2000_2022_3deg/fields.zarr",
		help="Output zarr short",
	)
	args = parser.parse_args()


	start_regridding(args.input, args.out, args.out_short)
