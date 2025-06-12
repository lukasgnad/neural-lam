from argparse import ArgumentParser
import xarray as xr
import gcsfs
import os
from neural_lam import constants as const
import time 
  
def append_era5_years_to_existing_zarr(zarr_path_gcs: str,
                                        local_zarr_path: str,
                                        start_date: str,
                                        end_date: str):
    """
    Downloads ERA5 Zarr data for the specified period and appends it to an existing local Zarr store.
    """
    start_time = time.time()

    print(f"Accessing GCS Zarr dataset from: {zarr_path_gcs}")
    fs = gcsfs.GCSFileSystem(token='anon')
    ds_remote = xr.open_zarr(fs.get_mapper(zarr_path_gcs), consolidated=True)

    print(f"Selecting data from {start_date} to {end_date}...")
    ds_subset = ds_remote.sel(time=slice(start_date, end_date))

    # Select atmospheric variables at specified pressure levels
    atm_subset = ds_subset[const.ATMOSPHERIC_PARAMS].sel(level=const.PRESSURE_LEVELS)
    
    # Select surface variables (no pressure level dimension)
    surface_vars = const.SURFACE_PARAMS + const.PREPROCESSING_PARAMS
    surf_subset = ds_subset[surface_vars]

    # Merge selected variables
    subset = xr.merge([atm_subset, surf_subset])

    print(f"Appending to existing local Zarr store at: {local_zarr_path}")
    subset.to_zarr(local_zarr_path, mode="a", append_dim="time", consolidated=True)

    elapsed_time = time.time() - start_time  # ⏱️ Stop the timer
    print(f"✅ Append complete. Elapsed time: {elapsed_time:.2f} seconds.")
     
def download_era5_subset_reduced(zarr_path_gcs: str,
	local_output_dir: str,
	start_date: str,
	end_date: str):
	"""
	Downloads a subset of ERA5 Zarr data from WeatherBench2 and saves it locally.
	Only selected variables and pressure levels are saved.
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
	# Select surface variables (no pressure level dimension)
	surf_subset = ds[surface_vars]

	# Merge selected variables
	subset = xr.merge([atm_subset, surf_subset])

	os.makedirs(local_output_dir, exist_ok=True)
	local_zarr_path = os.path.join(local_output_dir, "fields.zarr")

	print(f"Saving to local path: {local_zarr_path}")
	subset.to_zarr(local_zarr_path, consolidated=True, mode="w")

	elapsed_time = time.time() - start_time  # ⏱️ Stop the timer
	print(f"✅ Download and save complete. Elapsed time: {elapsed_time:.2f} seconds.")

def print_example(LOCAL_ZARR_PATH):
    # Open the Zarr store
	ds = xr.open_zarr(LOCAL_ZARR_PATH, consolidated=True)

	print(ds.dims)
	# Print the available time range
	print("Time range in dataset:")
	print(ds["time"].values[[0, -1]])  # first and last time values

	# Select a small subset from 2023 and 2024 for checking
	subset = ds.sel(time=slice("2023-01-01", "2024-12-31"))

	# Print available variables
	print("\nAvailable variables:")
	print(list(ds.data_vars))

	# Print example data for a single variable (e.g., temperature at 500 hPa)
	example_var = "geopotential_at_surface"  # or another variable from your dataset
	if example_var in subset:
		# Select for one time and location for demonstration
		example_data = subset[example_var].isel(time=0)
		print(f"\nExample data for {example_var} at first time step of 2023:")
		print(example_data)
	else:
		print(f"\nVariable '{example_var}' not found in dataset.")

	# Optionally, print times to confirm completeness
	print("\nExample times in 2023 and 2024:")
	print(subset["time"].values[:5])  # first few times
	print(subset["time"].values[-5:])  # last few times
    
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
	ZARR_PATH_GCS = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"


	""" MISSING_PERIODS = [
		("2023-01-01", "2024-12-31")
	]

	for start, end in MISSING_PERIODS:
		append_era5_years_to_existing_zarr(ZARR_PATH_GCS, LOCAL_OUTPUT_DIR + "/fields.zarr", start, end) """

	download_era5_subset_reduced(ZARR_PATH_GCS, args.out, args.start, args.end)
	# print_example(LOCAL_OUTPUT_DIR + "/fields.zarr")
