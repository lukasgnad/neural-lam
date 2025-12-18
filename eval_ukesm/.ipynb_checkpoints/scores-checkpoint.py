from argparse import ArgumentParser
import xarray as xr
from matplotlib import pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys

sys.path.insert(
    1, "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/evaluation"
)
import plot_utils as plot
import numpy as np


def create_folder_if_not_exists(folder):
    from pathlib import Path

    Path(folder).mkdir(parents=True, exist_ok=True)


def get_temporal_index_in_dataset(timestamp, ds):
    """
    Returns the integer index of a given timestamp in an xarray Dataset
    that may use non-Gregorian calendars (e.g., 360_day).

    timestamp: cftime.Datetime object or a string parseable by cftime
    ds: xarray.Dataset or DataArray with a 'time' coordinate
    """
    import cftime

    # Iterate over time coordinates to find match
    for i, t in enumerate(ds.time.values):
        if t == timestamp:
            return i

    # If not found, raise error
    raise ValueError(f"Timestamp {timestamp} not found in dataset time axis.")


def get_true_data_matching_forecast_timestamps(
    ds_truth, ds_forecast, prediction_steps
):
    """
    prediction_steps should be 0 based -> 0 means using the first prediction of the model for that timestamp
    """
    first_timestamp_forecast = ds_forecast.time[0].values
    last_timestamp_forecast = ds_forecast.time[-1].values
    idx_truth_first = (
        get_temporal_index_in_dataset(first_timestamp_forecast, ds_truth)
        + prediction_steps
        + 1
    )
    idx_truth_last = (
        get_temporal_index_in_dataset(last_timestamp_forecast, ds_truth)
        + prediction_steps
        + 1
    )
    new_forecast = ds_forecast.copy().isel(
        prediction_timedelta=prediction_steps
    )
    new_truth = ds_truth.isel(
        time=slice(idx_truth_first, idx_truth_last + 1)
    ).isel(time=slice(0, None, 2))
    new_forecast = new_forecast.sortby("time")
    _, index = np.unique(new_forecast["time"], return_index=True)
    new_forecast = new_forecast.isel(time=np.sort(index))

    for t in range(len(new_forecast.time.values) - 1):
        if (
            not str(
                new_forecast.time.values[t + 1] - new_forecast.time.values[t]
            )
            == "12:00:00"
        ):
            print(
                f"Error with {new_forecast.time.values[t]} and {new_forecast.time.values[t+1]}"
            )
    new_forecast = new_forecast.assign_coords({"time": new_truth["time"]})
    return new_truth, new_forecast


def get_stacked_true_data_matching_forecast_timestamps(ds_truth, ds_forecast):
    """
    prediction_steps should be 0 based -> 0 means using the first prediction of the model for that timestamp
    """
    first_timestamp_forecast = ds_forecast.time[0].values
    last_timestamp_forecast = ds_forecast.time[-1].values
    idx_truth_first = get_temporal_index_in_dataset(
        first_timestamp_forecast, ds_truth
    )
    idx_truth_last = get_temporal_index_in_dataset(
        last_timestamp_forecast, ds_truth
    )
    new_forecast = ds_forecast.copy()
    new_truth = ds_truth.isel(
        time=slice(idx_truth_first, idx_truth_last + 1)
    ).isel(time=slice(0, None, 2))
    new_forecast = new_forecast.sortby("time")
    _, index = np.unique(new_forecast["time"], return_index=True)
    new_forecast = new_forecast.isel(time=np.sort(index))
    new_forecast = new_forecast.chunk({"latitude": -1, "longitude": -1})
    new_truth = new_truth.chunk({"latitude": -1, "longitude": -1})
    # for t in range(len(new_forecast.time.values)-1):
    #     if not str(new_forecast.time.values[t+1] - new_forecast.time.values[t]) == '12:00:00':
    #         print(f'Error with {new_forecast.time.values[t]} and {new_forecast.time.values[t+1]}')
    # new_forecast = new_forecast.assign_coords({'time': new_truth['time']})
    return new_truth, new_forecast


def build_truth_with_leadtime(truth_ds, max_lead_steps=40):
    """
    truth_ds: Dataset with dimension (time, lat, lon, level)
    Creates a dataset with dimension (time, prediction_timedelta, lat, lon, level)

    lead step k uses truth_ds shifted by k steps forward in time.
    """

    lead_times = np.arange(1, max_lead_steps + 1)

    # Build stacked truth array
    truth_list = []
    for k in lead_times:
        truth_shifted = truth_ds.shift(
            time=-k
        )  # shift backwards so truth[t,k] = truth[t+k]
        truth_list.append(truth_shifted)

    # Combine along new dimension
    truth_with_lead = xr.concat(truth_list, dim="prediction_timedelta")

    # Assign actual timedelta
    truth_with_lead = truth_with_lead.assign_coords(
        prediction_timedelta=xr.DataArray(
            (lead_times * np.timedelta64(6, "h")).astype("timedelta64[ns]"),
            dims=("prediction_timedelta",),
        )
    )

    # Drop times near the end that became NaN
    truth_with_lead = truth_with_lead.dropna(dim="time", how="any")

    return truth_with_lead


def compute_r2_vectorized(model_idx=0, past_int=0):

    past = False if past_int == 0 else True

    model_resolution = [
        "z64",
        "z128",
        "z256",
        "z64 hierarchical",
        "persistence",
    ]

    current_model_resolution = model_resolution[model_idx]
    print(
        f'Calculating r2 scores for model {current_model_resolution} in {"past" if past else "future"}'
    )

    selected_model_past = plot.PAST_MODELS_UKESM[model_idx]
    selected_model_future = plot.FUTURE_MODELS_UKESM[model_idx]
    print(f'Evaluating model: {selected_model_past["model_name"]}')

    base_save_folder_past = selected_model_past["plot_folder"]
    base_save_folder_future = selected_model_future["plot_folder"]

    ukesm_past_forecasts = xr.open_zarr(
        selected_model_past["forecasts"], decode_timedelta=True
    ).astype(np.float32)
    ukesm_future_forecasts = xr.open_zarr(
        selected_model_future["forecasts"], decode_timedelta=True
    ).astype(np.float32)

    ukesm_past_truth = xr.open_zarr(selected_model_past["ground_truth"]).astype(
        np.float32
    )
    ukesm_future_truth = xr.open_zarr(
        selected_model_future["ground_truth"]
    ).astype(np.float32)

    create_folder_if_not_exists(base_save_folder_past)
    create_folder_if_not_exists(base_save_folder_future)

    print("Stacking truth data...")
    truth_stacked = build_truth_with_leadtime(
        ukesm_past_truth if past else ukesm_future_truth
    )
    print("Selecting matching forecasts and truth...")
    truth_ds, forecast_ds = get_stacked_true_data_matching_forecast_timestamps(
        truth_stacked, ukesm_past_forecasts if past else ukesm_future_forecasts
    )
    print("loading mask...")
    lat = truth_ds["latitude"]
    weights = np.cos(np.deg2rad(lat))
    weights = xr.DataArray(weights, coords={"latitude": lat}, dims=["latitude"])

    # normalize weights so they sum to 1 for stability
    weights = weights / weights.sum()
    all_r2 = {}
    mask = (
        xr.open_dataset("valid_masks.zarr")["past_test"]
        if past
        else xr.open_dataset("valid_masks.zarr")["future_test"]
    )

    for key in range(len(plot.KEY_VARIABLES_UKESM)):
        display_name, unit, variable, plevel = plot.KEY_VARIABLES_UKESM[key]
        print(f"Current var: {display_name}")
        # --- Select variable ---
        truth = truth_ds[variable]
        forecast = forecast_ds[variable]

        # --- Level selection ---
        if plevel is not None:
            truth = truth.sel(level=plevel)
            forecast = forecast.sel(level=plevel)

            if plevel > 500:
                mask_sel = mask.sel(level=plevel)
                truth = truth.where(mask_sel)
                forecast = forecast.where(mask_sel)

        truth_masked = truth
        forecast_masked = forecast

        # --- Broadcast weights ---
        # weights(lat) → (lat, lon)
        W2d = weights.broadcast_like(
            truth_masked.isel(time=0, prediction_timedelta=0)
        )

        # W2d → (time, lead, lat, lon)
        W = W2d.broadcast_like(truth_masked)

        # --- Weighted mean for each lead time ---
        # dims: ('time','lat','lon')
        truth_mean = (truth_masked * W).sum(
            dim=("latitude", "longitude")
        ) / W.sum(dim=("latitude", "longitude"))

        # truth_mean shape: (time, prediction_timedelta)

        # --- SSE & SST ---
        # subtract truth_mean along time dimension
        diff_truth = truth_masked - truth_mean

        # sum over time, lat, lon
        sse = ((truth_masked - forecast_masked) ** 2 * W).sum(
            dim=("time", "latitude", "longitude")
        )
        sst = ((diff_truth) ** 2 * W).sum(dim=("time", "latitude", "longitude"))

        r2 = 1 - sse / sst

        # Convert xarray DataArray → numpy vector
        r2_values = r2.values  # shape (40,)
        all_r2[variable] = r2_values

        import pandas as pd

        df_r2 = pd.DataFrame(all_r2).T  # variables = rows, steps = columns
        df_r2.columns = [f"step_{i+1}" for i in range(40)]

        df_r2.to_csv(
            "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/values/"
            + f"r2_all_variables_{current_model_resolution}_{'past' if past else 'future'}_new.csv"
        )


def compute_rmse_vectorized(model_idx=0, past_int=0):

    past = False if past_int == 0 else True

    model_resolution = [
        "z64",
        "z128",
        "z256",
        "z64 hierarchical",
        "persistence",
    ]

    current_model_resolution = model_resolution[model_idx]
    print(
        f'Calculating RMSE for model {current_model_resolution} in {"past" if past else "future"}'
    )

    selected_model_past = plot.PAST_MODELS_UKESM[model_idx]
    selected_model_future = plot.FUTURE_MODELS_UKESM[model_idx]
    print(f'Evaluating model: {selected_model_past["model_name"]}')

    base_save_folder_past = selected_model_past["plot_folder"]
    base_save_folder_future = selected_model_future["plot_folder"]

    ukesm_past_forecasts = xr.open_zarr(
        selected_model_past["forecasts"], decode_timedelta=True
    ).astype(np.float32)
    ukesm_future_forecasts = xr.open_zarr(
        selected_model_future["forecasts"], decode_timedelta=True
    ).astype(np.float32)

    ukesm_past_truth = xr.open_zarr(selected_model_past["ground_truth"]).astype(
        np.float32
    )
    ukesm_future_truth = xr.open_zarr(
        selected_model_future["ground_truth"]
    ).astype(np.float32)

    create_folder_if_not_exists(base_save_folder_past)
    create_folder_if_not_exists(base_save_folder_future)

    print("Stacking truth...")
    truth_stacked = build_truth_with_leadtime(
        ukesm_past_truth if past else ukesm_future_truth
    )

    print("Selecting matching truth / forecasts...")
    truth_ds, forecast_ds = get_stacked_true_data_matching_forecast_timestamps(
        truth_stacked, ukesm_past_forecasts if past else ukesm_future_forecasts
    )

    print("Preparing weights...")
    lat = truth_ds["latitude"]
    weights = np.cos(np.deg2rad(lat))
    weights = xr.DataArray(weights, coords={"latitude": lat}, dims=["latitude"])
    weights = weights / weights.sum()

    print("Loading mask...")
    mask = (
        xr.open_dataset("valid_masks.zarr")["past_test"]
        if past
        else xr.open_dataset("valid_masks.zarr")["future_test"]
    )

    all_rmse = {}
    rmse_maps = {}

    for key in range(len(plot.KEY_VARIABLES_UKESM)):

        display_name, unit, variable, plevel = plot.KEY_VARIABLES_UKESM[key]
        print(f"Current var (RMSE): {display_name}")

        # Select variable
        truth = truth_ds[variable]
        forecast = forecast_ds[variable]

        # Level selection
        if plevel is not None:
            truth = truth.sel(level=plevel)
            forecast = forecast.sel(level=plevel)

            if plevel > 500:
                truth = truth.where(mask.sel(level=plevel))
                forecast = forecast.where(mask.sel(level=plevel))

        # Broadcast weights
        W2d = weights.broadcast_like(truth.isel(time=0, prediction_timedelta=0))
        W = W2d.broadcast_like(truth)

        # Weighted MSE
        mse = ((forecast - truth) ** 2 * W).sum(
            dim=("time", "latitude", "longitude")
        ) / W2d.sum(dim=("latitude", "longitude"))

        rmse = np.sqrt(mse)
        
        
        mse_map = ((forecast - truth) ** 2).mean(dim="time")
        rmse_map = np.sqrt(mse_map)  # (prediction_timedelta, lat, lon)
        
        # 🎯 convention: use RMSE at the last lead time (40 steps)
        if plevel:
            rmse_maps[variable] = rmse_map.squeeze("level", drop=True)
        else:
            rmse_maps[variable] = rmse_map
        
        print(f"RMSE map for {variable}: {rmse_maps[variable].shape}")



        # Convert to numpy
        rmse_values = rmse.values  # shape (40,)
        all_rmse[variable] = rmse_values

        print(all_rmse[variable])
        # Save
        import pandas as pd

        df_rmse = pd.DataFrame(all_rmse).T
        df_rmse.columns = [f"step_{i+1}" for i in range(40)]

        save_path = (
            "/home/hk-project-pai00005/xo8179/neural_lam_fork/"
            "neural-lam/eval_ukesm/values/"
            f"rmse_all_variables_{current_model_resolution}_{'past' if past else 'future'}.csv"
        )

        df_rmse.to_csv(save_path)
        print("Saved:", save_path)

        ds_rmse_maps = xr.Dataset(
            {var: rmse_maps[var] for var in rmse_maps}
        )

        map_save = (
            "/home/hk-project-pai00005/xo8179/neural_lam_fork/"
            "neural-lam/eval_ukesm/values/"
            f"rmse_maps_{current_model_resolution}_{'past' if past else 'future'}.zarr"
        )
        ds_rmse_maps.to_zarr(map_save, mode="w")
        print("Saved RMSE maps:", map_save)


def main():
    """
    Pre-compute all static features related to the grid nodes
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="Model type (0=z64, 1=z128, 2=z256, 3=z64 hierarchical)"
        "(default: global_example_era5)",
    )
    parser.add_argument(
        "--past",
        type=int,
        default=0,
        help="0==future, 1==past",
    )
    parser.add_argument(
        "--score",
        type=str,
        default=0,
        help="rmse, r2",
    )

    args = parser.parse_args()

    if args.score == "r2":
        compute_r2_vectorized(args.model, args.past)
    elif args.score == "rmse":
        compute_rmse_vectorized(args.model, args.past)


if __name__ == "__main__":
    main()
