# Standard library
import os
from argparse import ArgumentParser

# Third-party
import graphcast.data_utils as gc_du
import solar_radiation_stable as gc_sr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cftime
import torch
import xarray as xa

# First-party
from neural_lam import vis  # type: ignore
from neural_lam.configs import get_constants

DEFAULT_DATASET = "global_example_era5"
DEFAULT_DATASET_PATH = "data"
DEFAULT_PLOT = 0
TIME_CHUNKS = 40


def get_year_progress(
    seconds_since_epoch: np.ndarray, dataset_type
) -> np.ndarray:
    """
    This was taken from the GraphCast repository and modified to support 360Day Year timestamps.
    The original code can be found here: https://github.com/google-deepmind/graphcast/blob/main/graphcast/data_utils.py

    Computes year progress for times in seconds.

    Args:
      seconds_since_epoch: Times in seconds since the "epoch" (the point at which
        UNIX time starts).

    Returns:
      Year progress normalized to be in the [0, 1) interval for each time point.
    """
    _SEC_PER_HOUR = 3600
    _HOUR_PER_DAY = 24
    SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
    _AVG_DAY_PER_YEAR = 365.24219
    _AVG_DAY_PER_YEAR_UKESM = 365

    # Start with the pure integer division, and then float at the very end.
    # We will try to keep as much precision as possible.
    if dataset_type == "ukesm":
        years_since_epoch = (
            seconds_since_epoch
            / SEC_PER_DAY
            / np.float64(_AVG_DAY_PER_YEAR_UKESM)
        )
    else:
        years_since_epoch = (
            seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
        )
    # Note depending on how these ops are down, we may end up with a "weak_type"
    # which can cause issues in subtle ways, and hard to track here.
    # In any case, casting to float32 should get rid of the weak type.
    # [0, 1.) Interval.
    return np.mod(years_since_epoch, 1.0).astype(np.float32)


def progress_to_sin_cos(progress):
    """
    Transform year/day progress in [0,1] with sin and cos, normalized to [0,1]
    """
    prog_sin = (np.sin(progress * 2 * np.pi) + 1) / 2
    prog_cos = (np.cos(progress * 2 * np.pi) + 1) / 2
    return prog_sin, prog_cos


def create_global_forcing(
    dataset: str = DEFAULT_DATASET,
    plot: int = DEFAULT_PLOT,
    dataset_path: str = DEFAULT_DATASET_PATH,
    dataset_type="era5",
):
    fields_group_path = os.path.join(dataset_path, dataset, "fields.zarr")
    fields_group = xa.open_zarr(fields_group_path)
    forcing_path = os.path.join(dataset_path, dataset, "forcing.zarr")

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
    print("Constructing timestamps")
    time_vals = fields_group.coords["time"].values

    # Handle CFTime vs datetime64 calendars
    if np.issubdtype(time_vals.dtype, np.datetime64):
        # Regular datetime64 (ERA5-style)
        seconds_since_epoch = timestamps.astype("int64")
        timestamps = fields_group.coords["time"].data.astype("datetime64[s]")
    else:
        # CFTime (360-day or other non-Gregorian calendars)
        timestamps = time_vals  # keep as cftime.Datetime360Day objects
        # Convert to numeric seconds manually
        epoch = cftime.Datetime360Day(1959, 1, 1, 0, 0, 0)
        seconds_since_epoch = np.array(
            [(t - epoch).total_seconds() for t in timestamps], dtype=np.float64
        )

    num_time = seconds_since_epoch.shape[0]

    # Create zarr to save to
    forcing_field_shape = (num_time, num_lon, num_lat)
    forcing_fields_dict = {}

    # TOA radiation
    print("Generating TOA radiation")
    toa_array = gc_sr.get_toa_incident_solar_radiation(
        timestamps,
        grid_lat_vals,
        grid_lon_vals,
    )  # (num_time, num_lat, num_lon)
    # Normalize to [0,1]
    toa_min = toa_array.min()
    toa_max = toa_array.max()
    toa_array = (toa_array - toa_min) / (toa_max - toa_min)
    forcing_fields_dict["toa_incident_radiation"] = toa_array.transpose(0, 2, 1)

    # Year progress
    print("Generating day + year progress features")
    year_progress = get_year_progress(
        seconds_since_epoch, dataset_type=dataset_type
    )
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
    print("Saving xarray")
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
        .chunk(
            {
                "time": TIME_CHUNKS,
                "longitude": -1,
                "latitude": -1,
                "forcing_var": -1,
            }
        )
    )
    xa_da.to_zarr(forcing_path, mode="w")
    print("Done!")

    if plot:
        # (num_vars, num_time, num_lon, num_lat)
        for time_i, timestamp in enumerate(timestamps):
            time_slice = xa_da.isel(time=time_i)  # (num_lon, num_lat, num_vars)

            for var_name in time_slice.coords["forcing_var"].data:
                forcing_field_xa = time_slice.sel(forcing_var=var_name)
                forcing_field = torch.tensor(
                    forcing_field_xa.to_numpy(), dtype=torch.float32
                ).flatten()
                vis.plot_prediction(
                    forcing_field,
                    forcing_field,
                    title=f"{timestamp} UTC, {var_name}",
                    const=get_constants(dataset_type),
                )
                plt.show()


def main():
    """
    Pre-compute all static features related to the grid nodes
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset to create grid features for "
        "(default: global_example_era5)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=DEFAULT_PLOT,
        help="If fields should be plotted " "(default: 0 (false))",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="The path to the folder containing the dataset (default 'data')",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="",
        help="The type of dataset: era5, nextgems, ukesm",
    )

    args = parser.parse_args()

    assert not args.dataset_type == "", f"No dataset_type given!"
    assert args.dataset_type in (
        "era5",
        "nextgems",
        "ukesm",
    ), f"Unknown dataset type: {args.dataset_type}"

    create_global_forcing(
        args.dataset, args.plot, args.dataset_path, args.dataset_type
    )


if __name__ == "__main__":
    main()
