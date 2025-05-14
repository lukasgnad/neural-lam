# Standard library
import os

# Third-party
import numpy as np
import torch
import xarray as xa

# First-party
from neural_lam import constants, utils

class ERA5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name,
        pred_length=40,
        split="train",
        standardize=True,
        expanded_test=False,
        **kwarg,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), "Unknown dataset split"

        fields_path = os.path.join("data", dataset_name, "fields.zarr")
        fields_xds = xa.open_zarr(fields_path)
        forcing_path = os.path.join("data", dataset_name, "forcing.zarr")
        forcing_xda = xa.open_dataarray(forcing_path, engine="zarr")

        # Custom split logic for your available date range (2018–2020)
        split_slices = {
            "train": slice("2018-01-01T12", "2019-12-31T12"),
            "val": slice("2019-01-01T00", "2019-12-31T12"),
            "test": slice("2020-01-01T00", "2020-12-31T12"),
        }

        fields_ds_split = fields_xds.sel(time=split_slices[split])
        forcing_ds_split = forcing_xda.sel(time=split_slices[split])

        timesteps_in_split = len(fields_ds_split.coords["time"])
        self.pred_length = pred_length
        ds_timesteps = timesteps_in_split - 1 - pred_length

        # Modified assertion
        if ds_timesteps <= 0:
            print(f"⚠️ WARNING: Not enough data in '{split}' split for pred_length={pred_length}. "
                  f"Timesteps available: {timesteps_in_split}")
            self.ds_len = 0
        else:
            if split == "train":
                self.ds_len = ds_timesteps
                self.init_all = True
            else:
                self.ds_len = int(np.ceil(ds_timesteps / 2))
                self.init_all = False

        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean = ds_stats["data_mean"]
            self.data_std = ds_stats["data_std"]

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
        self.atm_total_dim = len(self.atm_xda.coords["level"]) * len(self.atm_xda.coords["state_var"])
        self.surface_total_dim = len(self.surface_xda.coords["state_var"])

        self.forcing_xda = forcing_ds_split

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        if self.ds_len == 0:
            raise IndexError("Dataset is empty due to insufficient time range.")

        if self.init_all:
            init_i = idx + 1
        else:
            init_i = 1 + idx * 2

        sample_slice = slice(init_i - 1, init_i + self.pred_length + 1)
        full_series_len = self.pred_length + 2

        atm_sample_np = self.atm_xda[sample_slice].to_numpy()
        surface_sample_np = self.surface_xda[sample_slice].to_numpy()

        full_state_np = np.concatenate(
            (
                atm_sample_np.reshape((full_series_len, -1, self.atm_total_dim)),
                surface_sample_np.reshape((full_series_len, -1, self.surface_total_dim)),
            ),
            axis=-1,
        )

        full_state_torch = torch.tensor(full_state_np, dtype=torch.float32)
        if self.standardize:
            full_state_torch = (full_state_torch - self.data_mean) / self.data_std

        init_states = full_state_torch[:2]
        target_states = full_state_torch[2:]

        forcing_np = self.forcing_xda[sample_slice].to_numpy()
        forcing_flat_np = forcing_np.reshape(full_series_len, -1, forcing_np.shape[-1])
        forcing_windowed = np.concatenate(
            (forcing_flat_np[:-2], forcing_flat_np[1:-1], forcing_flat_np[2:]),
            axis=2,
        )

        forcing_torch = torch.tensor(forcing_windowed, dtype=torch.float32)

        return init_states, target_states, forcing_torch
