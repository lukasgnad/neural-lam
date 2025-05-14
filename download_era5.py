import xarray as xr
import gcsfs
import os
  
def download_era5_subset(
	zarr_path_gcs: str,
	local_output_dir: str,
	start_date: str,
	end_date: str
	):
	"""
	Downloads a subset of ERA5 Zarr data from WeatherBench2 and saves it locally.
	  
	Parameters:
	zarr_path_gcs (str): GCS path to Zarr dataset.
	local_output_dir (str): Local directory where data will be saved.
	start_date (str): Start date in 'YYYY-MM-DD' format.
	end_date (str): End date in 'YYYY-MM-DD' format.
	"""
	print(f"Accessing GCS Zarr dataset from: {zarr_path_gcs}")
	fs = gcsfs.GCSFileSystem(token='anon')
	ds = xr.open_zarr(fs.get_mapper(zarr_path_gcs), consolidated=True)
	  
	print(f"Selecting data from {start_date} to {end_date}...")
	subset = ds.sel(time=slice(start_date, end_date))
	  
	os.makedirs(local_output_dir, exist_ok=True)
	local_zarr_path = os.path.join(local_output_dir, "fields.zarr")
	  
	print(f"Saving to local path: {local_zarr_path}")
	subset.to_zarr(local_zarr_path, consolidated=True, mode="w")
	  
	print("✅ Download and save complete.")

if __name__ == "__main__":
    # Example usage: download 2018–2020 data at 5.625° resolution
    ZARR_PATH_GCS = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    EXAMPLE = False
    # This fetches data for a small sample run of the library
    if EXAMPLE:
        LOCAL_OUTPUT_DIR = "data/example_global_era5"
        START_DATE = "1959-01-01"
        END_DATE = "1959-01-05"
    # This fetches data for the exact test period used in the library
    else:
        LOCAL_OUTPUT_DIR = "data/global_era5_train001"
        START_DATE = "2018-12-31"
        END_DATE = "2020-01-11"

    download_era5_subset(ZARR_PATH_GCS, LOCAL_OUTPUT_DIR, START_DATE, END_DATE)
