import xarray as xr
from matplotlib import pyplot as plt
import xarray as xr
from scipy.stats import wasserstein_distance
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plot_utils as plot
import numpy as np


def weighted_quantile(values, quantile, sample_weight=None):
    """Compute weighted quantile (0–1)."""
    sorter = np.argsort(values)
    values, weights = values[sorter], sample_weight[sorter]
    cdf = np.cumsum(weights) / np.sum(weights)
    return np.interp(quantile, cdf, values)


UNICODE_MINUS = "\u2212"


def format_shift(value, unit=""):
    """Dynamically format small shift values with appropriate precision."""
    abs_val = abs(value)
    if abs_val >= 0.1:
        fmt = f"{value:+.2f}"  # ±0.12
    elif abs_val >= 0.01:
        fmt = f"{value:+.3f}"  # ±0.012
    elif abs_val >= 0.001:
        fmt = f"{value:+.4f}"  # ±0.0012
    else:
        fmt = f"{value:+.1e}"  # scientific notation for really tiny shifts
    fmt = fmt.replace("-", UNICODE_MINUS)
    return f"{fmt}{unit}"


def format_value(value, unit=""):
    """Dynamically format small shift values with appropriate precision."""
    abs_val = abs(value)
    if abs_val >= 0.1:
        fmt = f"{value:.2f}"  # ±0.12
    elif abs_val >= 0.01:
        fmt = f"{value:.3f}"  # ±0.012
    elif abs_val >= 0.001:
        fmt = f"{value:.4f}"  # ±0.0012
    else:
        fmt = f"{value:.1e}"  # scientific notation for really tiny shifts
    fmt = fmt.replace("-", UNICODE_MINUS)
    return f"{fmt}{unit}"


def calc_single_dist_with_nans(obs_1, lat_name="latitude", log_x=False):

    # --- Compute cosine(lat) weights ---
    lats_1 = obs_1[lat_name].values
    w_lat_1 = np.cos(np.deg2rad(lats_1))
    w_1 = np.broadcast_to(
        w_lat_1[None, None, :], obs_1.shape
    )  # shape (time, lat, lon)

    # --- Flatten values and weights ---
    vals1 = obs_1.values.flatten()

    w1 = w_1.flatten()

    # --- Remove NaNs safely ---
    mask1 = np.isfinite(vals1)
    vals1 = vals1[mask1]
    w1 = w1[mask1]
    # --- Weighted means ---
    mean1 = np.average(vals1, weights=w1)
    # --- Weighted 95th percentiles ---
    p95_1 = weighted_quantile(vals1, 0.95, sample_weight=w1)
    p99_1 = weighted_quantile(vals1, 0.99, sample_weight=w1)
    p99_9_1 = weighted_quantile(vals1, 0.999, sample_weight=w1)
    p5_1 = weighted_quantile(vals1, 0.05, sample_weight=w1)

    if log_x:
        vals1 = vals1 + 1e-6

    return {
        "vals": vals1,
        "weights": w1 / w1.sum(),
        "mean": mean1,
        "p95": p95_1,
        "p99": p99_1,
        "p999": p99_9_1,
        "p5": p5_1,
    }


def plot_distributions_with_nans_on_axis(
    ax,
    obs,
    background,
    bin_num=50,
    variable_name="Variable",
    unit="",
    title="",
    plot_legend=True,
    colors=None,
    plot_5_quantile=False,
    log_x=False,
    log_y=False,
    plot_99_quant=False,
):
    """
    Plot weighted histograms of distributions for obs_2019 vs obs_2049.
    Handles NaNs safely after masking.

    Parameters
    ----------
    obs : xarray.DataArray
        Input data with dims (time, lat, lon)
    lat_name : str
        Name of the latitude coordinate
    bins : int
        Number of bins for the histogram
    variable_name : str
        Label for x-axis
    unit : str
        Unit string for labels
    save_to : str or None
        File path to save the figure (if None, show interactively)
    """
    if not colors:
        colors = [f"C{i}" for i in range(len(obs))]
    for i in range(len(obs)):
        obs[i]["color"] = colors[i]

    mean = []
    p95 = []
    # --- Add vertical lines ---
    bins = bin_num

    def draw_lines(
        data,
        reference=None,
        print_difference=False,
        plot_5_quantile=False,
        plot_legend=True,
    ):
        label = data['label'] + ' ' if not data['label'] == '' else ''
        if not print_difference:
            
            mean_l = (
                None# f"{label}mean ({format_value(data['mean'], unit)})"
                if plot_legend
                else None
            )
            if plot_99_quant:
                p95_l = (
                    None#f"{label}99.9th % ({format_value(data['p999'], unit)})"
                    if plot_legend
                    else None
                )
            else:
                p95_l = (
                    None#f"{label}95th % ({format_value(data['p95'], unit)})"
                    if plot_legend
                    else None
                )
            p5_l = (
                None#f"{label}5th % ({format_value(data['p5'], unit)})"
                if plot_legend
                else None
            )
        else:
            mean_l = (
                f"{label}mean ({reference['label']}{format_shift(obs[i]['mean'] - obs[0]['mean'], unit)})"
                if plot_legend
                else f"{reference['label']}{format_shift(obs[i]['mean'] - obs[0]['mean'], unit)}"
            )
            if plot_99_quant:
                p95_l = (
                    f"{label}99.9th % ({reference['label']}{format_shift(obs[i]['p999'] - obs[0]['p999'], unit)})"
                    if plot_legend
                    else f"{reference['label']}{format_shift(obs[i]['p999'] - obs[0]['p999'], unit)}"
                )
            else:
                p95_l = (
                    f"{label}95th % ({reference['label']}{format_shift(obs[i]['p95'] - obs[0]['p95'], unit)})"
                    if plot_legend
                    else f"{reference['label']}{format_shift(obs[i]['p95'] - obs[0]['p95'], unit)}"
                )
            p5_l = (
                f"{label}5th % ({reference['label']}{format_shift(obs[i]['p5'] - obs[0]['p5'], unit)})"
                if plot_legend
                else f"{reference['label']}{format_shift(obs[i]['p5'] - obs[0]['p5'], unit)}"
            )
        ax.axvline(
            data["mean"],
            color=data["color"],
            linestyle=(0, (1, 0)),
            linewidth=2,
            label=mean_l,
        )
        ax.axvline(
            data["p999"] if plot_99_quant else data["p95"],
            color=data["color"],
            linestyle=(0, (1, 3)),
            linewidth=2,
            label=p95_l,
        )
        if plot_5_quantile:
            ax.axvline(
                data["p5"],
                color=data["color"],
                linestyle=(0, (5, 3)),
                linewidth=2,
                label=p5_l,
            )

    # if plot_legend:
    for i in range(len(background)):
        if "hist" in background[i].keys():
            ax.hist(
                background[i]["bins"][:-1],
                background[i]["bins"],
                weights=background[i]["hist"],
                alpha=0.5,
                label=background[i]["label_full"] if plot_legend else None,
                color="grey",
            )
        else:
            ax.hist(
                background[i]["vals"],
                bins=bins,
                weights=background[i]["weights"],
                alpha=0.5,
                label=background[i]["label_full"] if plot_legend else None,
                density=True,
                color="grey",
            )
    for i in range(len(obs)):
        if log_x:
            bins = np.logspace(
                np.log10(1e-6), np.log10(obs[i]["vals"].max()), bin_num
            )
        if "hist" in obs[i].keys():
            ax.hist(
                obs[i]["bins"][:-1],
                obs[i]["bins"],
                weights=obs[i]["hist"],
                alpha=0.5,
                label=obs[i]["label_full"] if plot_legend else None,
                color=obs[i]["color"],
            )
        else:
            ax.hist(
                obs[i]["vals"],
                bins=bins,
                weights=obs[i]["weights"],
                alpha=0.5,
                label=obs[i]["label_full"] if plot_legend else None,
                density=True,
                color=obs[i]["color"],
            )
        draw_lines(
            obs[i],
            reference=obs[0],
            print_difference=not (i == 0),
            plot_5_quantile=plot_5_quantile,
            plot_legend=plot_legend,
        )

    # --- Labels and layout ---
    ax.set_xlabel(f"{variable_name} [{unit}]" if unit else variable_name)
    if not log_x:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))
    # ax.locator_params(axis='x', nbins=5)
    ax.xaxis.get_offset_text().set_fontsize(8)
    if plot_legend:
        ax.set_ylabel("Probability density")
    if log_x:
        ax.set_xscale("log")
    # ax.set_yscale('log')
    ax.set_title(title)
    s = variable_name.lower()
    if "humidity" in s or "precip" in s:
        ax.legend(loc="upper right")
    else:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_distributions_with_nans(
    obs,
    lat_name="latitude",
    bins=50,
    variable_name="Variable",
    unit="",
    save_to=None,
    title="",
    labels=[],
    colors=None,
    plot_5_quantile=False,
    display_name="",
    plot_size=(7, 6),
):
    """
    Plot weighted histograms of distributions for obs_2019 vs obs_2049.
    Handles NaNs safely after masking.

    Parameters
    ----------
    obs_1, obs_2 : xarray.DataArray
        Input data with dims (time, lat, lon)
    lat_name : str
        Name of the latitude coordinate
    bins : int
        Number of bins for the histogram
    variable_name : str
        Label for x-axis
    unit : str
        Unit string for labels
    save_to : str or None
        File path to save the figure (if None, show interactively)
    """
    if "precip" in variable_name.lower():
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=plot_size)
        plot_distributions_with_nans_on_axis(
            ax=axes,
            obs=obs,
            background=[],
            bin_num=bins,
            variable_name=variable_name,
            unit=unit,
            colors=colors,
            title=title,
            plot_5_quantile=plot_5_quantile,
            plot_99_quant=True,
        )
        axes.set_yscale("log")
        if save_to:
            plt.savefig(
                save_to.replace(".png", "_y_log.png").replace(
                    ".pdf", "_y_log.pdf"
                ),
                bbox_inches="tight",
                dpi=300,
            )
        else:
            plt.show()
        bins = 1200

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=plot_size)
    plot_distributions_with_nans_on_axis(
        ax=axes,
        obs=obs,
        background=[],
        bin_num=bins,
        variable_name=variable_name,
        unit=unit,
        colors=colors,
        title=title,
        plot_5_quantile=plot_5_quantile,
    )
    if "precip" in variable_name.lower():
        xmin, xmax = axes.get_xlim()
        row_xlim = (xmin, obs[0]["p99"])
        axes.set_xlim(row_xlim)
        axes.set_xlabel(
            f"{display_name} [{unit}], cut off at {obs[0]['label']} 99% ({obs[0]['p99']:.2f}{unit})"
            if unit
            else f"{display_name}, cut of at GT 99% ({obs[0]['p99']:.2f})"
        )
    else:
        axes.set_xlim(get_xlim(variable_name))
    if save_to:
        plt.savefig(save_to, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def get_xlim(variable, dataset="both"):
    if dataset.lower() == "ng":
        if "u-component" in variable.lower():
            return (-90, 40)
        if "v-component" in variable.lower():
            return (-70, 30)
        if "vorticity" in variable.lower():
            return (-3.5e-4, 1.5e-4)
        if "humidity" in variable.lower():
            return (-0.001, 0.015)
        if "geopotential" in variable.lower():
            return (4.5e4, 6e4)
        if "temperature" in variable.lower() and "850" in variable.lower():
            return (220, 312)
        elif "temperature" in variable.lower():
            return (205, 325)
    if dataset.lower() == "ukesm_wide":
        if "u-component" in variable.lower():
            return (-70, 40)
        if "v-component" in variable.lower():
            return (-50, 30)
        if "vorticity" in variable.lower():
            return (-2.5e-4, 1.5e-4)
        if "humidity" in variable.lower():
            return (-0.001, 0.016)
        if "geopotential" in variable.lower():
            return (4.6e4, 6.1e4)
        if "temperature" in variable.lower() and "850" in variable.lower():
            return (220, 320)
        elif "temperature" in variable.lower():
            return (210, 330)
    if dataset.lower() == "ng_wide":
        if "u-component" in variable.lower():
            return (-70, 40)
        if "v-component" in variable.lower():
            return (-50, 30)
        if "humidity" in variable.lower():
            return (-0.001, 0.015)
        if "geopotential" in variable.lower():
            return (4.5e4, 6e4)
        if "temperature" in variable.lower() and "850" in variable.lower():
            return (220, 312)
        elif "temperature" in variable.lower():
            return (205, 325)
    if "u-component" in variable.lower():
        return (-90, 40)
    if "v-component" in variable.lower():
        return (-70, 30)
    if "vorticity" in variable.lower():
        return (-3.5e-4, 1.5e-4)
    if "humidity" in variable.lower():
        return (-0.001, 0.0175)
    if "geopotential" in variable.lower():
        return (4.5e4, 6.1e4)
    if "temperature" in variable.lower() and "850" in variable.lower():
        return (220, 320)
    elif "temperature" in variable.lower():
        return (190, 330)


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
        if not (
            str(new_forecast.time.values[t + 1] - new_forecast.time.values[t])
            == "12:00:00"
            or new_forecast.time.values[t + 1] - new_forecast.time.values[t]
            == 43200000000000
        ):
            print(
                f"Error with {new_forecast.time.values[t]} and {new_forecast.time.values[t+1]}\n    Difference is {str(new_forecast.time.values[t+1] - new_forecast.time.values[t])}"
            )
    new_forecast = new_forecast.assign_coords({"time": new_truth["time"]})
    return new_truth, new_forecast


def get_datasets_after(
    ds_true, ds_forecast, hours_into_future=None, days_into_future=None
):
    steps = -1
    if days_into_future:
        steps += 4 * days_into_future
    if hours_into_future:
        if not hours_into_future % 6 == 0:
            print(
                f"    Warning, given hours: {hours_into_future} cannot be divided by step length (6)"
            )
        steps += int(hours_into_future / 6)
    if steps == -1:
        steps = 0
        print(
            f"    No valid length provided, setting steps={steps} (=={(steps+1)*6} hours)"
        )
    else:
        print(
            f"    Starting with a step length of {steps} (=={(steps+1)*6} hours)"
        )
    return get_true_data_matching_forecast_timestamps(
        ds_true, ds_forecast, steps
    )


DEFAULT_DIST_SAVE_FOLDER = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_nextgems_new/distributions"


def precompute_and_save_distribution(
    data,
    variable,
    dataset="N",
    dataslice_name="TR",
    is_ground_truth=True,
    save_folder=DEFAULT_DIST_SAVE_FOLDER,
    bins=500,
    rollout=0,
    model_id="",
    lower_bound=False,
):
    if lower_bound:
        # Filter data before histogram
        print(f'Lowest number: {np.nanmin(data["vals"])}')
        mask = data["vals"] >= lower_bound
        vals = data["vals"][mask]
        weights = data["weights"][mask]
        total_weight = data["weights"].sum()
        kept_weight = weights.sum()

        removed_pct = 100 * (1 - kept_weight / total_weight)
        print(f"Percentage removed: {removed_pct:.5f}%")
    else:
        vals = data["vals"]
        weights = data["weights"]

    hist, edges = np.histogram(vals, bins=bins, weights=weights, density=True)
    data["hist"] = hist
    data["bins"] = edges

    data_new = {}
    for key in data.keys():
        if key == "vals" or key == "weights":
            continue
        data_new[key] = data[key]

    if not save_folder[-1] == "/":
        save_folder = save_folder + "/"
    save_name = f'{dataset}_{dataslice_name}/{"GT" if is_ground_truth else "FC"}_{variable}{"" if rollout==0 else f"_{rollout}"}{"" if model_id=="" else f"_{model_id}"}.npz'

    np.savez(save_folder + save_name, **data_new)


def load_dist(
    variable,
    dataset="N",
    dataslice_name="TR",
    is_ground_truth=True,
    save_folder=DEFAULT_DIST_SAVE_FOLDER,
    rollout=0,
    model_id="",
):
    if not save_folder[-1] == "/":
        save_folder = save_folder + "/"
    save_name = f'{dataset}_{dataslice_name}/{"GT" if is_ground_truth else "FC"}_{variable}{"" if rollout==0 else f"_{rollout}"}{"" if model_id=="" else f"_{model_id}"}.npz'
    print(f"loading {save_name}")
    loaded = np.load(save_folder + save_name)
    new_dict = {}
    for key in loaded.keys():
        new_dict[key] = loaded[key]
    return new_dict
