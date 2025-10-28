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

from neural_lam.configs import get_constants  # type: ignore

# First-party
from neural_lam import utils


def parse_periods(periods_str):
    periods = {}
    if periods_str is None:
        return periods
    for period in periods_str.split(";"):
        name, dates = period.split(":")
        start, end = dates.split(",")
        periods[name] = slice(start, end)
    return periods


# FIX for UKESM 360day years
def parse_periods(periods_str, use_cftime=False):
    """
    Parses a string like 'train:1990-01-01,2010-12-30;test:2015-01-01,2019-12-30'
    into a dict of slices, using either datetime64 or CFTime objects.
    """
    periods = {}
    if periods_str is None:
        return periods

    for period in periods_str.split(";"):
        name, dates = period.split(":")
        start, end = dates.split(",")

        if use_cftime:
            # Convert to cftime objects safely
            def safe_cftime(date_str, is_end=False):
                import cftime

                y, m, d = map(int, date_str.split("-"))
                # Clamp day to 30 if needed (for 360-day calendars)
                d = min(d, 30)
                if is_end:
                    return cftime.Datetime360Day(y, m, d, 23)
                return cftime.Datetime360Day(y, m, d)

            start = safe_cftime(start)
            end = safe_cftime(end, True)
        else:
            # Regular datetime64 timestamps
            start = np.datetime64(start)
            end = np.datetime64(end)

        periods[name] = slice(start, end)

    return periods


class ERA5Dataset(torch.utils.data.Dataset):
    """
    Dataset loading ERA5 from Zarr
    """

    def __init__(
        self,
        dataset_name,
        periods="",
        pred_length=40,
        split="train",
        standardize=True,
        dataset_path="data",
        dataset_type="era5",
        **kwarg,  # pylint: disable=unused-argument
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"

        C = get_constants(dataset_type)

        # Open xarrays
        fields_path = os.path.join(dataset_path, dataset_name, "fields.zarr")
        fields_xds = xa.open_zarr(fields_path)
        forcing_path = os.path.join(dataset_path, dataset_name, "forcing.zarr")
        forcing_xda = xa.open_dataarray(forcing_path, engine="zarr")
        # each with dims (num_time, num_lon, num_lat)

        # FIX for UKESM 360day years
        use_cftime = not np.issubdtype(fields_xds.time.dtype, np.datetime64)
        split_slices = parse_periods(periods, use_cftime=use_cftime)
        # split_slices = parse_periods(periods)

        fields_ds_split = fields_xds.sel(time=split_slices[split])
        forcing_ds_split = forcing_xda.sel(time=split_slices[split])

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
        if standardize:
            ds_stats = utils.load_dataset_stats(
                dataset_name, dataset_path, "cpu"
            )

            # These are torch arrays
            self.data_mean = ds_stats["data_mean"]
            self.data_std = ds_stats["data_std"]

        # Turn into directly indexable Dataarrays
        # Fields, in order
        self.atm_xda = (
            fields_ds_split[C.ATMOSPHERIC_PARAMS]
            .to_dataarray("state_var")
            .transpose("time", "longitude", "latitude", "state_var", "level")
        )
        self.surface_xda = (
            fields_ds_split[C.SURFACE_PARAMS]
            .to_dataarray("state_var")
            .transpose("time", "longitude", "latitude", "state_var")
        )
        # store dimensions for later reshaping
        self.atm_total_dim = len(self.atm_xda.coords["level"]) * len(
            self.atm_xda.coords["state_var"]
        )
        self.surface_total_dim = len(self.surface_xda.coords["state_var"])
        # Do final flattening of levels and stacking in __getitem__ with numpy

        # Forcing, already a Dataarray of correct shape
        self.forcing_xda = forcing_ds_split
        # (num_time, num_lon, num_lat, forcing_dim)

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
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

        return init_states, target_states, forcing_torch
