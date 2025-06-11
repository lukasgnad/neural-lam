# Standard library
import os
from argparse import ArgumentParser

# Third-party
import graphcast.losses as gc_l
import numpy as np
import torch
import xarray as xa
import gcsfs
from tqdm import tqdm

# First-party
from neural_lam import constants
from neural_lam.era5_dataset_old import ERA5Dataset
from neural_lam.weather_dataset import WeatherDataset

DEFAULT_DATASET= "meps_example"
DEFAULT_BATCH_SIZE=32
DEFAULT_STEP_LENGTH =3
DEFAULT_N_WORKERS=4
DEFAULT_DATASET_PATH="data"
DEFAULT_ZARR="weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

def create_parameter_weights(dataset:str=DEFAULT_DATASET, batch_size:int=DEFAULT_BATCH_SIZE,
                             step_length:int=DEFAULT_STEP_LENGTH, n_workers:int=DEFAULT_N_WORKERS,
                             dataset_path:str=DEFAULT_DATASET_PATH, zarr_path=DEFAULT_ZARR):
    static_dir_path = os.path.join(dataset_path, dataset, "static")
    if not os.path.exists(static_dir_path):
        os.makedirs(static_dir_path)
    global_ds = "global" in dataset

    if global_ds:
        # Follow approach of GraphCast, giving vertical levels weight
        # proportional to pressure, and hand-design for surface vars
        pres_levels_np = np.array(constants.PRESSURE_LEVELS, dtype=np.float32)
        # Weighting for one variable at all pressure levels sum to 1
        pres_levels_norm = pres_levels_np / pres_levels_np.sum()  # (num_vert,)
        atm_weights = np.tile(
            pres_levels_norm, len(constants.ATMOSPHERIC_PARAMS)
        )  # (num_atm * num_vert,)

        surface_weights = np.array(
            [
                1.0 if var_name == "2t" else 0.1
                for var_name in constants.SURFACE_PARAMS_SHORT
            ],
            dtype=np.float32,
        )  # (num_surf,)
        vert_weights = np.concatenate((atm_weights, surface_weights), axis=0)
        # (num_variables,)

        # Compute spatial weighting for grid nodes
        print(f"Accessing GCS Zarr dataset from: {zarr_path}")
        fs = gcsfs.GCSFileSystem(token='anon')
        xds = xa.open_zarr(fs.get_mapper(zarr_path), consolidated=True)
        # fields_group_path = os.path.join(dataset_path, dataset, "fields.zarr")
        # xds = xa.open_zarr(fields_group_path)
        # Hack since GC code uses "lat" for some reason
        xds = xds.assign_coords({"lat": xds.coords["latitude"]})

        lat_weights = gc_l.normalized_latitude_weights(xds)
        lat_weights_torch = torch.tensor(
            lat_weights.to_numpy(), dtype=torch.float32
        )
        num_lon = len(xds.coords["longitude"])
        grid_weights = (
            lat_weights_torch.unsqueeze(0).repeat(num_lon, 1).flatten()
        )
        torch.save(
            grid_weights, os.path.join(static_dir_path, "grid_weights.pt")
        )
    else:
        # Create parameter weights based on height
        # based on fig A.1 in graph cast paper
        w_dict = {
            "2": 1.0,
            "0": 0.1,
            "65": 0.065,
            "1000": 0.1,
            "850": 0.05,
            "500": 0.03,
        }
        vert_weights = np.array(
            [w_dict[par.split("_")[-2]] for par in constants.PARAM_NAMES_SHORT],
            dtype=np.float32,
        )
    print("Saving parameter weights...")
    np.save(
        os.path.join(static_dir_path, "parameter_weights.npy"), vert_weights
    )

    # Load dataset without any subsampling
    if global_ds:
        ds = ERA5Dataset(
            dataset,
            split="train",
            pred_length=1,  # Use 1 to get each time step only once
            standardize=False,
            dataset_path=dataset_path
        )
    else:
        ds = WeatherDataset(
            dataset,
            split="train",
            subsample_step=1,
            pred_length=63,
            standardize=False,
        )  # Without standardization
    loader = torch.utils.data.DataLoader(
        ds, batch_size, shuffle=False, num_workers=n_workers
    )
    # Compute mean and std.-dev. of each parameter (+ flux forcing)
    # across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    flux_means = []
    flux_squares = []
    for init_batch, target_batch, forcing_batch in tqdm(loader):
        if global_ds:
            batch = target_batch  # (N_batch, N_t=1, N_grid, d_features)
        else:
            batch = torch.cat(
                (init_batch, target_batch), dim=1
            )  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(
            torch.mean(batch**2, dim=(1, 2))
        )  # (N_batch, d_features,)

        if not global_ds:
            # Flux at 1st windowed position is index 1 in forcing
            flux_batch = forcing_batch[:, :, :, 1]
            flux_means.append(torch.mean(flux_batch))  # (,)
            flux_squares.append(torch.mean(flux_batch**2))  # (,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    print("Saving mean, std.-dev, flux_stats...")
    torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))

    if not global_ds:
        flux_mean = torch.mean(torch.stack(flux_means))  # (,)
        flux_second_moment = torch.mean(torch.stack(flux_squares))  # (,)
        flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
        flux_stats = torch.stack((flux_mean, flux_std))
        torch.save(flux_stats, os.path.join(static_dir_path, "flux_stats.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    # Re-load dataset with standardization
    if global_ds:
        ds_standard = ERA5Dataset(
            dataset,
            split="train",
            pred_length=1,  # Use 1 to get each time step only once
            standardize=True,
            dataset_path=dataset_path
        )
    else:
        ds_standard = WeatherDataset(
            dataset,
            split="train",
            subsample_step=1,
            pred_length=63,
            standardize=True,
        )
        used_subsample_len = (65 // step_length) * step_length
    loader_standard = torch.utils.data.DataLoader(
        ds_standard, batch_size, shuffle=False, num_workers=n_workers
    )

    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _ in tqdm(loader_standard):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t', N_grid, d_features)

        if global_ds:
            # Only extract state at init time and target at next time
            stepped_batch = batch[:, 1:]  # (N_batch, 2, N_grid, d_features)
        else:
            # Note: batch contains only 1h-steps
            stepped_batch = torch.cat(
                [
                    batch[:, ss_i : used_subsample_len : step_length]
                    for ss_i in range(step_length)
                ],
                dim=0,
            )
            # (N_batch', N_t, N_grid, d_features),
            # N_batch' = step_length*N_batch

        batch_diffs = stepped_batch[:, 1:] - stepped_batch[:, :-1]
        # (N_batch', N_t-1, N_grid, d_features)

        diff_means.append(
            torch.mean(batch_diffs, dim=(1, 2))
        )  # (N_batch', d_features,)
        diff_squares.append(
            torch.mean(batch_diffs**2, dim=(1, 2))
        )  # (N_batch', d_features,)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
    torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))

def main():
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset to compute weights for (default: meps_example)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=DEFAULT_STEP_LENGTH,
        help="Step length in hours to consider single time step (for LAM only)"
        " (default: 3)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="The path to the folder containing the dataset (default \'data\')",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=DEFAULT_ZARR,
        help="Zarr path to load grid point coordinates from "
        "(default: global_example_era5)",
    )
    args = parser.parse_args()
    create_parameter_weights(dataset=args.dataset, batch_size=args.batch_size, step_length=args.step_length,
                             n_workers=args.n_workers, dataset_path=args.dataset_path, zarr_path=args.zarr_path)
    

import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()
