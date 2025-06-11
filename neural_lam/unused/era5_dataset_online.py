# Standard library
import os

# Third-party
import numpy as np
import torch
import graphcast.data_utils as gc_du
import graphcast.solar_radiation as gc_sr
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
import gcsfs
import xarray as xa




# First-party
from neural_lam import constants, utils

def progress_to_sin_cos(progress):
    """
    Transform year/day progress in [0,1] with sin and cos, normalized to [0,1]
    """
    prog_sin = (np.sin(progress * 2 * np.pi) + 1) / 2
    prog_cos = (np.cos(progress * 2 * np.pi) + 1) / 2
    return prog_sin, prog_cos


def create_global_forcing(fields_group, toa_min_max):
    

    # Lat-lon
    grid_lat_vals = np.array(
        fields_group["latitude"], dtype=np.float32
    )  # (num_lat,)
    grid_lon_vals = np.array(
        fields_group["longitude"], dtype=np.float32
    )  # (num_long,)
    num_lat = grid_lat_vals.shape[0]
    num_lon = grid_lon_vals.shape[0]

    # Construct timestamps
    # Time 0 here is 1959-01-01, 00:00
    timestamps = fields_group.coords["time"].data.astype("datetime64[s]")

    # Number of seconds since unix time (can be negative)
    seconds_since_epoch = timestamps.astype(np.int64)
    num_time = seconds_since_epoch.shape[0]

    # Create zarr to save to
    forcing_field_shape = (num_time, num_lon, num_lat)
    forcing_fields_dict = {}

    # TOA radiation
    toa_array = gc_sr.get_toa_incident_solar_radiation(
        timestamps,
        grid_lat_vals,
        grid_lon_vals,
    )  # (num_time, num_lat, num_lon)
    # Normalize to [0,1]
    toa_array = (toa_array - toa_min_max[0]) / (toa_min_max[1] - toa_min_max[0])
    forcing_fields_dict["toa_incident_radiation"] = toa_array.transpose(0, 2, 1)

    # Year progress
    year_progress = gc_du.get_year_progress(seconds_since_epoch)
    # (num_time,)
    year_prog_sin, year_prog_cos = progress_to_sin_cos(year_progress)
    forcing_fields_dict["sin_year_progress"] = np.broadcast_to(
        year_prog_sin[:, np.newaxis, np.newaxis], forcing_field_shape
    )
    forcing_fields_dict["cos_year_progress"] = np.broadcast_to(
        year_prog_cos[:, np.newaxis, np.newaxis], forcing_field_shape
    )

    # Day progress
    # Note that this is slightly off as GC only uses a similar modulo calc.
    day_progress = gc_du.get_day_progress(
        seconds_since_epoch, grid_lon_vals
    )  # (num_time, num_lon)
    day_prog_sin, day_prog_cos = progress_to_sin_cos(day_progress)
    forcing_fields_dict["sin_day_progress"] = np.broadcast_to(
        day_prog_sin[:, :, np.newaxis], forcing_field_shape
    )
    forcing_fields_dict["cos_day_progress"] = np.broadcast_to(
        day_prog_cos[:, :, np.newaxis], forcing_field_shape
    )

    # Save as xarray stored with zarr
    coord_names = ("time", "longitude", "latitude")
    xa_ds = xa.Dataset(
        {
            var_name: (coord_names, var_vals)
            for var_name, var_vals in forcing_fields_dict.items()
        },
        coords={coord: fields_group.coords[coord] for coord in coord_names},
    )
    xa_da = (
        xa_ds.to_dataarray("forcing_var")
        .transpose("time", "longitude", "latitude", "forcing_var")
        .chunk({"time": 1, "longitude": -1, "latitude": -1, "forcing_var": -1})
    )
    return xa_da



class ERA5Dataset(torch.utils.data.Dataset):
    """
    Dataset loading ERA5 from Zarr
    """

    def __init__(
        self,
        dataset_name,
        pred_length=40,
        split="train",
        standardize=True,
        expanded_test=False,
        dataset_path="data",
        **kwarg,  # pylint: disable=unused-argument
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        # Open xarrays
        # fields_path = os.path.join(dataset_path, dataset_name, "fields.zarr")
        # fields_xds = xa.open_zarr(fields_path)
        fs = gcsfs.GCSFileSystem(token='anon')
        zarr_path = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
        fields_xds = xa.open_zarr(fs.get_mapper(zarr_path), consolidated=True)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.initialized = False
        # each with dims (num_time, num_lon, num_lat)

        # Slice to split into train / val / test
        if "example" in dataset_name:
            # Example subset, create some example split
            split_slices = {
                "train": slice("1959-01-01T12", "1959-01-03T12"),  # 3 days
                "val": slice("1959-01-03T18", "1959-01-04T18"),  # 1 day
                "test": slice("1959-01-03T18", "1959-01-04T18"),  # Same as val
            }
        elif "train001" in dataset_name:
            # Example subset, create some example split
            print("Splitting for train001...")
            split_slices = {
                "train": slice("2019-01-01T12", "2019-09-30T12"),  # 3 days
                "val": slice("2019-10-01T12", "2019-10-31T12"),  # 1 day
                "test": slice("2019-11-01T12", "2019-12-31T18"),  # Same as val
            }
        elif "train003" in dataset_name:
            # Example subset, create some example split
            print("Splitting for train003...")
            split_slices = {
                "train": slice("2019-01-01T12", "2019-09-30T12"),  # 3 days
                "val": slice("2019-10-01T12", "2019-10-31T12"),  # 1 day
                "test": slice("2019-11-01T12", "2019-12-31T18"),  # Same as val
            }
        else:
            # Actual dataset
            # Note that we start at 12 on first day as first two timesteps have
            # NaN for precipitation
            split_slices = {
                "train": slice("1959-01-01T12", "2017-12-31T12"),  # 1959-2017
                "val": slice("2017-12-31T18", "2019-12-31T12"),  # 2018-2019
            }
            if expanded_test:
                # 2020-2023
                split_slices["test"] = slice("2019-12-31T18", "2023-12-31T18")
            else:
                # 2020 only, consistent with WB2 (forecasts extend into 2021)
                # Extend 40 time steps into 2021
                split_slices["test"] = slice("2019-12-31T18", "2021-01-10T18")

        fields_ds_split = fields_xds.sel(time=split_slices[split])
        self.split_slice = split_slices[split]

        # Compute dataset length
        timesteps_in_split = len(fields_ds_split.coords["time"])
        self.pred_length = pred_length
        
        # -1 for AR-2, - pred_length for target states
        ds_timesteps = timesteps_in_split - 1 - pred_length
        assert ds_timesteps > 0, "Dataset too small for given pred_length"
        if split == "train":
            # Init from all timesteps
            self.ds_len = ds_timesteps
            self.init_all = True
        else:  # val, test
            # Init only form 00/12 UTC
            self.ds_len = int(np.ceil(ds_timesteps / 2))
            self.init_all = False

        # Set up for standardization
        self.standardize = standardize
        if self.standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, dataset_path, "cpu")

            # These are torch arrays
            self.data_mean = ds_stats["data_mean"]
            self.data_std = ds_stats["data_std"]

    def _lazy_init(self):
        # Turn into directly indexable Dataarrays
        # Fields, in order
        fs = gcsfs.GCSFileSystem(token='anon')
        zarr_path = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
        fields_xds = xa.open_zarr(fs.get_mapper(zarr_path), consolidated=True)
        fields_ds_split = fields_xds.sel(time=self.split_slice)
        self.atm_xda = (
            fields_ds_split[constants.ATMOSPHERIC_PARAMS]
            .to_dataarray("state_var")
            .transpose("time", "longitude", "latitude", "state_var", "level")
        )
        self.surface_xda = (
            fields_ds_split[constants.SURFACE_PARAMS]
            .to_dataarray("state_var")
            .transpose("time", "longitude", "latitude", "state_var")
        )
        # store dimensions for later reshaping
        self.atm_total_dim = len(self.atm_xda.coords["level"]) * len(
            self.atm_xda.coords["state_var"]
        )
        self.surface_total_dim = len(self.surface_xda.coords["state_var"])
        # Do final flattening of levels and stacking in __getitem__ with numpy

        forcing_path = os.path.join(self.dataset_path, self.dataset_name, "forcing.zarr")
        forcing_xda = xa.open_dataarray(forcing_path, engine="zarr")
        forcing_ds_split = forcing_xda.sel(time=self.split_slice)
        # Forcing, already a Dataarray of correct shape
        self.forcing_xda = forcing_ds_split
        # (num_time, num_lon, num_lat, forcing_dim)
        self.initialized = True

    
    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        try:
            import time
            start_time = time.time()
            if not self.initialized:
                if idx == 0:
                    print("Starting lazy init")
                self._lazy_init()
                if idx == 0:
                    print("Ending lazy init")
            # Forecast t=(s+1):(s+pred_length) from init states at t=s-1,s
            if self.init_all:
                init_i = idx + 1  # s = idx+1
            else:
                # Only initialize at 00/12 UTC timesteps
                init_i = 1 + idx * 2  # s = 1 + 2idx
            sample_slice = slice(init_i - 1, init_i + self.pred_length + 1)
            full_series_len = self.pred_length + 2

            # === Sample ===
            # Extract and stack sample fields from zarr
            atm_sample_np = self.atm_xda[sample_slice].to_numpy()
            # (2+pred_length, num_lon, num_lat, d_atm, num_levels)
            surface_sample_np = self.surface_xda[sample_slice].to_numpy()
            # (2+pred_length, num_lon, num_lat, d_surface)

            full_state_np = np.concatenate(
                (
                    atm_sample_np.reshape(
                        (full_series_len, -1, self.atm_total_dim)
                    ),  # (2+pred_length, num_grid, d_atm')
                    surface_sample_np.reshape(
                        (full_series_len, -1, self.surface_total_dim)
                    ),  # (2+pred_length, num_grid, d_surface)
                ),
                axis=-1,
            )  # (2+pred_length, num_grid, state_dim)

            # Convert to torch
            full_state_torch = torch.tensor(full_state_np, dtype=torch.float32)
            if self.standardize:
                # Standardize sample
                full_state_torch = (
                    full_state_torch - self.data_mean
                ) / self.data_std

            # Split into init_states and target
            init_states = full_state_torch[:2]
            target_states = full_state_torch[2:]

            # === Forcing features ===
            # Note that forcing should be sliced for same length, first and last
            # time steps will be eaten up by windowing
            # Extract forcing from zarr
            forcing_np = self.forcing_xda[sample_slice].to_numpy()
            # forcing_np = create_global_forcing(self.atm_xda[sample_slice], self.toa_min_max).to_numpy()
            # (2+pred_length, num_lon, num_lat, forcing_dim)

            # Flatten lat-lon dim
            forcing_flat_np = forcing_np.reshape(
                full_series_len, -1, forcing_np.shape[-1]
            )  # (pred_length, num_grid, forcing_dim)

            # Window and stack 3 time steps
            forcing_windowed = np.concatenate(
                (
                    forcing_flat_np[:-2],
                    forcing_flat_np[1:-1],
                    forcing_flat_np[2:],
                ),
                axis=2,
            )  # (pred_length, num_grid, forcing_dim')

            # Convert to torch tensor
            forcing_torch = torch.tensor(forcing_windowed, dtype=torch.float32)
            # Do not need to standardize forcing, already handled in generation
            # === Measure and log memory size (only once)
            if idx == 0:                                                                                                
                def tensor_size_MB(tensor):
                    return tensor.element_size() * tensor.nelement() / (1024 ** 2)

                total_MB = (
                    tensor_size_MB(init_states)
                    + tensor_size_MB(target_states)
                    + tensor_size_MB(forcing_torch)
                )
                print(f"[INFO] Sample size at idx=0: {total_MB:.2f} MB")  
                
                elapsed_time = time.time() - start_time  # ⏱️ Stop the timer
                print(f"[INFO] Fetched data at idx=0 in {elapsed_time:.2f} seconds.")                      

            return init_states, target_states, forcing_torch
        except Exception as e:
                print(f"Worker error: {e}")
                raise e
