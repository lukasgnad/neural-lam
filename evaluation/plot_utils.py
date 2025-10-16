import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib import pyplot as plt

KEY_VARIABLES = [
    ("Geopotential at 500hPa", "m^2/s^2", "geopotential", 500),
    ("Temperature at 850hPa", "K", "temperature", 850),
    ("Specific humidity at 700hPa", "kg/kg", "specific_humidity", 700),
    ("T2m", "K", "2m_temperature", None),
    ("U-component of wind at 850hPa", "m/s", "u_component_of_wind", 850),
    ("V-component of wind at 850hPa", "m/s", "v_component_of_wind", 850),
    ("Total precipiation (6hr)", "m", "total_precipitation_6hr", None),
]


base_path = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/"
PAST_MODELS = [
    # ERA5 tested on NextGEMS data
    {
        "model_name": "ERA5 z64 multilevel graph",
        "folder": [
            base_path
            + f"era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_0{run_number}"
            for run_number in [1, 2, 3]
        ],
        "folder_long": [
            base_path
            + f"era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_0{run_number}_long"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "train_time": ["6h 53m 50s", "4h 11m 9s", "7h 33m 57s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z64)
    {
        "model_name": "NextGEMS z64 multilevel graph",
        "folder": [
            base_path
            + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_0{run_number}"
            for run_number in [1, 2, 3]
        ],
        "folder_long": [
            base_path
            + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_0{run_number}_long"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "train_time": ["6h 55m 40s", "4h 13m 48s", "7h 52m 48s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z128)
    {
        "model_name": "NextGEMS z128 multilevel graph",
        "folder": [
            base_path
            + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_0{run_number}"
            for run_number in [1, 2, 3]
        ],
        "folder_long": [
            base_path
            + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_0{run_number}_long"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "train_time": ["10h 43m 33s", "6h 17m 45s", "10h 42m 10s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z256)
    {
        "model_name": "NextGEMS z256 multilevel graph",
        "folder": [
            None
        ],
        "folder_long": [
            base_path
            + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/test_runs/wandb_test_0{run_number}_long"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "train_time": ["16h 53m 16s", "10h 33m 14s", "18h 10m 39s"],
    },
    # Nextgems conservative hierarchical (1990-2020, 128x64, z64)
    {
        "model_name": "NextGEMS z64 hierarchical graph",
        "folder": [
            base_path
            + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/test_runs/wandb_test_0{run_number}"
            for run_number in [1, 2, 3]
        ],
        "folder_long": [
            base_path
            + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/test_runs/wandb_test_0{run_number}_long"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "train_time": ["23h 3m 26s", "15h 23m 25s", "27h 28m 57s"],
    },
    # Nextgems Persistence conservative
    {
        "model_name": "Persistence",
        "folder": [
            base_path
            + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_persistence"
            for run_number in [1, 2, 3]
        ],
        "folder_long": [
            base_path
            + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_persistence_long"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "train_time": None,
    },
]

FUTURE_MODELS = [
    # ERA5 tested on NextGEMS data
    {
        "model_name": "ERA5 z64 multilevel graph",
        "folder": base_path
        + f"era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_03_2049",
        "folder_long": base_path
        + f"era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_03_2049_long",
        "mae": None,
        "rmse": None,
        "train_time": ["6h 53m 50s", "4h 11m 9s", "7h 33m 57s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z64)
    {
        "model_name": "NextGEMS z64 multilevel graph",
        "folder": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03_2049",
        "folder_long": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03_2049_long",
        "mae": None,
        "rmse": None,
        "train_time": ["6h 55m 40s", "4h 13m 48s", "7h 52m 48s"],
    },
    
    # Nextgems conservative multilevel (1990-2020, 128x64, z128)
    {
        "model_name": "NextGEMS z128 multilevel graph",
        "folder": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_03_2049",
        "folder_long": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_03_2049_long",
        "mae": None,
        "rmse": None,
        "train_time": ["10h 43m 33s", "6h 17m 45s", "10h 42m 10s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z256)
    {
        "model_name": "NextGEMS z256 multilevel graph",
        "folder": None,
        "folder_long": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/test_runs/wandb_test_03_2049_long",
        "mae": [],
        "rmse": [],
        "train_time": ["16h 53m 16s", "10h 33m 14s", "18h 10m 39s"],
    },
    # Nextgems conservative hierarchical (1990-2020, 128x64, z64)
    {
        "model_name": "NextGEMS z64 hierarchical graph",
        "folder": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/test_runs/wandb_test_03_2049",
        "folder_long": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/test_runs/wandb_test_03_2049_long",
        "mae": None,
        "rmse": None,
        "train_time": ["23h 3m 26s", "15h 23m 25s", "27h 28m 57s"],
    },
    # Nextgems Persistence conservative
    {
        "model_name": "Persistence",
        "folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_2049_persistence",
        "folder_long": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_2049_persistence_long",
        "mae": None,
        "rmse": None,
        "train_time": None,
    },
]


def plot_double_plot(
    data_left,
    title_left,
    cbar_label_left,
    vbounds_left,
    cmap_left,
    data_right,
    title_right,
    cbar_label_right,
    vbounds_right,
    cmap_right,
    base_save_folder,
    save_name,
):

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), subplot_kw={"projection": proj}
    )

    # --- Shift map ---
    if vbounds_left:
        im = data_left.plot(
            ax=axes[0],
            cmap=cmap_left,
            transform=proj,
            vmin=vbounds_left[0],
            vmax=vbounds_left[1],
            add_colorbar=False,
        )
    else:
        im = data_left.plot(
            ax=axes[0], cmap=cmap_left, transform=proj, add_colorbar=False
        )
    axes[0].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[0].add_feature(cfeature.BORDERS, linewidth=0.5)
    axes[0].set_title(title_left)
    cbar = plt.colorbar(im, ax=axes[0], shrink=0.5, aspect=30, pad=0.02)
    cbar.set_label(cbar_label_left)

    if vbounds_right:
        im = data_right.plot(
            ax=axes[1],
            cmap=cmap_right,
            transform=proj,
            vmin=vbounds_right[0],
            vmax=vbounds_right[1],
            add_colorbar=False,
        )
    else:
        im = data_right.plot(
            ax=axes[1], cmap=cmap_right, transform=proj, add_colorbar=False
        )
    axes[1].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[1].add_feature(cfeature.BORDERS, linewidth=0.5)
    axes[1].set_title(title_right)
    cbar = plt.colorbar(im, ax=axes[1], shrink=0.5, aspect=30, pad=0.02)
    cbar.set_label(cbar_label_right)

    plt.tight_layout()
    plt.savefig(
        base_save_folder + f"{save_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_single_plot(
    data_left,
    title_left,
    cbar_label_left,
    vbounds_left,
    cmap_left,
    base_save_folder,
    save_name,
):

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 1, figsize=(7, 5), subplot_kw={"projection": proj}
    )

    # --- Shift map ---
    if vbounds_left:
        im = data_left.plot(
            ax=axes,
            cmap=cmap_left,
            transform=proj,
            vmin=vbounds_left[0],
            vmax=vbounds_left[1],
            add_colorbar=False,
        )
    else:
        im = data_left.plot(
            ax=axes, cmap=cmap_left, transform=proj, add_colorbar=False
        )
    axes.add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes.add_feature(cfeature.BORDERS, linewidth=0.5)
    axes.set_title(title_left)
    cbar = plt.colorbar(im, ax=axes, shrink=0.5, aspect=30, pad=0.02)
    cbar.set_label(cbar_label_left)

    plt.tight_layout()
    plt.savefig(
        base_save_folder + f"{save_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


import matplotlib.pyplot as plt


def plot_zonal_plot(
    data,
    title,
    y_label,
    vbounds=None,
    base_save_folder="./",
    save_name="single_plot",
    label_above="Performance decrease in 2049",
    label_below="Performance increase in 2049",
    centered_around=1,
    color_above="red",
    color_below="green",
):
    """
    Plot a zonal (longitude-averaged) mean as a latitude profile.

    Parameters
    ----------
    data : xr.DataArray
        Input DataArray with dims (lat, lon) or (time, lat, lon).
    title : str
        Title of the plot.
    y_label : str
        Y-axis label.
    vbounds : tuple, optional
        (vmin, vmax) for y-axis limits.
    base_save_folder : str
        Folder to save the figure.
    save_name : str
        Output filename (without extension).
    """
    # if data has lon, average zonally
    if "longitude" in data.dims:
        zonal_mean = data.mean(dim="longitude")
    else:
        zonal_mean = data

    # average over time if present
    if "time" in zonal_mean.dims:
        zonal_mean = zonal_mean.mean(dim="time")

    lat = zonal_mean["latitude"].values
    vals = zonal_mean.values

    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot the line
    ax.plot(lat, vals, color="k", lw=1.5)

    # Reference line at 1
    ax.axhline(centered_around, color="gray", linestyle="--", lw=1)

    # Fill areas
    ax.fill_between(
        lat,
        centered_around,
        vals,
        where=(vals > centered_around),
        color=color_above,
        alpha=0.3,
        label=label_above,
    )
    ax.fill_between(
        lat,
        centered_around,
        vals,
        where=(vals < centered_around),
        color=color_below,
        alpha=0.3,
        label=label_below,
    )

    # Labels and formatting
    ax.set_title(title)
    ax.set_xlabel("Latitude")
    ax.set_ylabel(y_label)

    if vbounds:
        ax.set_ylim(vbounds[0], vbounds[1])
    else:
        max_val = max(abs(vals.min()), abs(vals.max())) * 1.2
        ax.set_ylim(centered_around - max_val, centered_around + max_val)

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        base_save_folder + f"{save_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
