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


import matplotlib.pyplot as plt


def plot_single_plot(
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
