
from argparse import ArgumentParser
import xarray as xr
import gcsfs
import os
import time
import numpy as np
import xesmf as xe
from neural_lam import constants as const


def save_var(var:str, da_regrid):
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
    return da_out


def download_era5_subset_reduced_regridded(zarr_path_gcs: str,
    local_output_dir: str,
    start_date: str,
    end_date: str):
    """
    Downloads a subset of ERA5 Zarr data from WeatherBench2,
    regrids it to 4° resolution, and saves it locally.
    """
    start_time = time.time()

    print(f"Accessing GCS Zarr dataset from: {zarr_path_gcs}")
    fs = gcsfs.GCSFileSystem(token='anon')
    ds = xr.open_zarr(fs.get_mapper(zarr_path_gcs), consolidated=True)

    print(f"Selecting data from {start_date} to {end_date}...")
    ds = ds.sel(time=slice(start_date, end_date))

    # Select atmospheric variables at specified pressure levels
    atm_subset = ds[const.ATMOSPHERIC_PARAMS].sel(level=const.PRESSURE_LEVELS)
    surface_vars = const.SURFACE_PARAMS + const.PREPROCESSING_PARAMS
    surf_subset = ds[surface_vars]

    # Merge selected variables
    subset = xr.merge([atm_subset, surf_subset])

    # Define coarse 3° grid
    target_grid = xr.Dataset({
        'lat': (['lat'], np.arange(-90, 90.1, 3.0)),
        'lon': (['lon'], np.arange(0, 360.0, 3.0))
    })

    print("Regridding to 3°x3° resolution...")
    # If 'subset' has multiple variables, we must handle vertical levels carefully

    regridded_vars = []

    # Create regridders for methods you need
    regridder_bilinear = xe.Regridder(ds, target_grid, "bilinear", periodic=True)
    regridder_conservative = xe.Regridder(ds, target_grid, "conservative", periodic=True)
    regridder_nearest = xe.Regridder(ds, target_grid, "nearest_s2d", periodic=True)

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
    if os.path.exists(local_output_dir):
        import shutil
        shutil.rmtree(local_output_dir)


    os.makedirs(local_output_dir, exist_ok=True)
    local_zarr_path = os.path.join(local_output_dir, "fields_3deg.zarr")
    first_save = True

    for var in subset.data_vars:
        da = subset[var]
        if 'level' in da.dims:
            # Loop through pressure levels
            for lev in da.level.values:
                if var in bilinear_vars:
                    da_regrid = regridder_bilinear(da)
                elif var in conservative_vars:
                    da_regrid = regridder_conservative(da)
                elif var in nearest_vars:
                    da_regrid = regridder_nearest(da)
                else:
                    print(f"Warning: {var} not assigned a regridding method, skipping.")
                    continue
                if first_save:
                    save_var(var, da_regrid).to_zarr(local_zarr_path, consolidated=True, mode="w")
                    first_save = False
                else:
                    save_var(var, da_regrid).to_zarr(local_zarr_path, consolidated=True, mode="a")
                print(f"Saved variable {var}, pressure level {lev}")
        else:
            if var in bilinear_vars:
                da_regrid = regridder_bilinear(da)
            elif var in conservative_vars:
                da_regrid = regridder_conservative(da)
            elif var in nearest_vars:
                da_regrid = regridder_nearest(da)
            else:
                print(f"Warning: {var} not assigned a regridding method, skipping.")
                continue
            if first_save:
                save_var(var, da_regrid).to_zarr(local_zarr_path, consolidated=True, mode="w")
                first_save = False
            else:
                save_var(var, da_regrid).to_zarr(local_zarr_path, consolidated=True, mode="a")
            print(f"Saved variable {var}")

    elapsed_time = time.time() - start_time
    print(f"✅ Download + regrid + save complete. Elapsed time: {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--out",
        type=str,
        default="/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_era5_1980_2022",
        help="Local output directory",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1980-01-01",
        help="Start date",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2022-12-31",
        help="End Date",
    )
    args = parser.parse_args()


    # Example usage: download 2018–2020 data
    # ZARR_PATH_GCS = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    ZARR_PATH_GCS = "weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

    download_era5_subset_reduced_regridded(ZARR_PATH_GCS, args.out, args.start, args.end)
    # print_example(LOCAL_OUTPUT_DIR + "/fields.zarr")
