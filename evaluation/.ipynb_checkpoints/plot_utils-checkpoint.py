from enum import Enum
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib import pyplot as plt

KEY_VARIABLES = [
    ("Geopotential at 500hPa", "m²/s²", "geopotential", 500, "Geopotential"),
    ("Temperature at 850hPa", "K", "temperature", 850, "Temperature"),
    (
        "Specific Humidity at 700hPa",
        "kg/kg",
        "specific_humidity",
        700,
        "Specific Humidity",
    ),
    ("2m Temperature", "K", "2m_temperature", None, "2m Temperature"),
    (
        "U-Component of Wind at 850hPa",
        "m/s",
        "u_component_of_wind",
        850,
        "U-Component of Wind",
    ),
    (
        "V-Component of Wind at 850hPa",
        "m/s",
        "v_component_of_wind",
        850,
        "V-Component of Wind",
    ),
    (
        "Total Precipitation",
        "m",
        "total_precipitation_6hr",
        None,
        "Total Precipitation",
    ),
]

KEY_VARIABLES_UKESM = [
    (
        "Geopotential at 500hPa",
        "m²/s²",
        "geopotential_500",
        None,
        "Geopotential",
    ),
    ("Temperature at 850hPa", "K", "temperature", 850, "Temperature"),
    (
        "Specific Humidity at 700hPa",
        "kg/kg",
        "specific_humidity",
        700,
        "Specific Humidity",
    ),
    ("1.5m Temperature", "K", "surface_temperature", None, "1.5m Temperature"),
    (
        "U-Component of Wind at 850hPa",
        "m/s",
        "u_component_of_wind",
        850,
        "U-Component of Wind",
    ),
    (
        "V-Component of Wind at 850hPa",
        "m/s",
        "v_component_of_wind",
        850,
        "V-Component of Wind",
    ),
    (
        "Total Precipitation",
        "m",
        "total_precipitation_6hr",
        None,
        "Total Precipitation",
    ),
    (
        "Rel. Vorticity at 500hPa",
        "1/s",
        "relative_vorticity",
        500,
        "Rel. Vorticity",
    ),
]

def get_multiplier_ukesm(var):
    if 'geopotential' in var.lower():
        return UKESM_GEOPOT_CONST
    elif 'precipitation' in var.lower():
        return UKESM_PRECIP_CONST
    else:
        return 1


UKESM_GEOPOT_CONST = 9.80665
UKESM_PRECIP_CONST = 0.001


class DATASET_SLICE(Enum):
    TRAIN = 0
    TEST_HIST = 1
    TEST_FUTURE = 2


def get_masked_ukesm_data(dataset, plevel):
    import xarray as xr

    mask = xr.open_dataset(
        "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/valid_masks.zarr"
    )["all_combined"].sel(level=plevel)

    dataset = dataset.reindex_like(mask, method="nearest", tolerance=1e-6)
    masked_data = dataset.where(mask)
    assert len(masked_data.latitude) == len(dataset.latitude)
    assert len(masked_data.longitude) == len(dataset.longitude)
    return masked_data


def sync_axes(axes: plt.Axes, sync_x=True, sync_y=True):
    xmins, xmaxs = [], []
    ymins, ymaxs = [], []
    for i, ax in enumerate(axes):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
    max_x_lim = (min(xmins), max(xmaxs))
    max_y_lim = (min(ymins), max(ymaxs))
    for i, ax in enumerate(axes):
        if sync_x:
            ax.set_xlim(max_x_lim)
        if sync_y:
            ax.set_ylim(max_y_lim)


def fix_lon_lat(ds):
    import xarray as xr

    mask = xr.open_dataset(
        "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/valid_masks.zarr"
    )["all_combined"].sel(level=850)
    ds = ds.reindex_like(mask, method="nearest", tolerance=1e-6)
    return ds


def open_zarr_ukesm(path: str, decode_timedelta=True):
    import xarray as xr

    return fix_lon_lat(
        xr.open_zarr(path, decode_timedelta=decode_timedelta).astype(np.float32)
    )


base_path = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/"
PAST_MODELS_NG_era5 = [
    # Nextgems conservative multilevel (1990-2020, 128x64, z64)
    {
        "model_name": "NextGEMS z64 Multi-Mesh",
        "id": "z64_past",
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
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only_forecasts/z64_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/plots/hist/",
        "train_time": ["6h 55m 40s", "4h 13m 48s", "7h 52m 48s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z128)
    {
        "model_name": "NextGEMS z128 Multi-Mesh",
        "id": "z128_past",
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
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only_forecasts/z128_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/plots/hist/",
        "train_time": ["10h 43m 33s", "6h 17m 45s", "10h 42m 10s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z256)
    {
        "model_name": "NextGEMS z256 Multi-Mesh",
        "id": "z256_past",
        "folder": [None],
        "folder_long": [
            base_path
            + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/test_runs/wandb_test_0{run_number}_long"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only_forecasts/z256_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/plots/hist/",
        "train_time": ["16h 53m 16s", "10h 33m 14s", "18h 10m 39s"],
    },
    # Nextgems conservative hierarchical (1990-2020, 128x64, z64)
    {
        "model_name": "NextGEMS z64 Hierarchical",
        "id": "z64 hierarchical_past",
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
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only_forecasts/z64_graph_fm.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/plots/hist/",
        "train_time": ["23h 3m 26s", "15h 23m 25s", "27h 28m 57s"],
    },
    # Nextgems Persistence conservative
    {
        "model_name": "Persistence",
        "id": "persistence_past",
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
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only_forecasts/z64_persistence.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/plots/hist_persistence/",
        "train_time": None,
    },
    # ERA5 tested on NextGEMS data
    {
        "model_name": "ERA5 z64 Multi-Mesh",
        "id": "z64_era5_past",
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
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only_forecasts/z64_graphcast_era5.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only/fields.zarr",
        "plot_folder": base_path + "era5_1990_2020_6h-128x64_z64/plots/hist/",
        "train_time": ["6h 53m 50s", "4h 11m 9s", "7h 33m 57s"],
    },
]


base_path_ukesm = base_path  # "/hkfs/work/workspace_haic/scratch/xo8179-neural_lam_copy/neural-lam/trained_models/"
PAST_MODELS_UKESM = [
    # UKESM conservative multilevel (1885-2014, 128x64, z64)
    {
        "model_name": "UKESM z64 Multi-Mesh",
        "id": "z64_past",
        "folder": [
            base_path_ukesm
            + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_0{run_number}_no_forecasts"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64/plots/hist/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_forecasts/z64_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks/fields.zarr",
        "train_time": ["6h 11m 32s", "5h 8m 49s", "6h 9m 42s"],
    },
    # UKESM conservative multilevel (1885-2014, 128x64, z128)
    {
        "model_name": "UKESM z128 Multi-Mesh",
        "id": "z128_past",
        "folder": [
            base_path_ukesm
            + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_0{run_number}_no_forecasts"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z128/plots/hist/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_forecasts/z128_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks/fields.zarr",
        "train_time": ["13h 56m 1s", "5h 41m 56s", "9h 57m 34s"],
    },
    # UKESM conservative multilevel (1885-2014, 128x64, z256)
    {
        "model_name": "UKESM z256 Multi-Mesh",
        "id": "z256_past",
        "folder": [
            base_path_ukesm
            + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256/test_runs/wandb_test_0{run_number}_no_forecasts"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256/plots/hist/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_forecasts/z256_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks/fields.zarr",
        "train_time": ["15h 45m 26s", "16h 26m 31s", "17h 30m 7s"],
    },
    # UKESM conservative hierarchical (1885-2014, 128x64, z64)
    {
        "model_name": "UKESM z64 Hierarchical",
        "id": "z64 hierarchical_past",
        "folder": [
            base_path_ukesm
            + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/test_runs/wandb_test_0{run_number}_no_forecasts"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/plots/hist/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_forecasts/z64_graph_fm.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks/fields.zarr",
        "train_time": ["25h 13m 59s", "17h 26m 43s", "26h 34m 13s"],
    },
    # UKESM Persistence conservative
    {
        "model_name": "Persistence",
        "id": "persistence_past",
        "folder": [
            base_path_ukesm
            + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_persistence"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64/plots/hist_persistence/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_forecasts/z64_persistence.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks/fields.zarr",
        "train_time": None,
    },
]

PAST_MODELS_UKESM_HOLES_FILLED = [
    # UKESM conservative multilevel (1885-2014, 128x64, z64)
    {
        "model_name": "UKESM z64 Multi-Mesh Interpolated",
        "id": "z64_past_interpolated",
        "folder": [
            base_path_ukesm
            + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64_holes_filled/test_runs/wandb_test_0{run_number}_no_forecasts"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64_holes_filled/plots/hist/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_holes_filled_forecasts/z64_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_holes_filled/fields.zarr",
        "train_time": ["6h 11m 32s", "5h 8m 49s", "6h 9m 42s"],
    },
    # UKESM conservative multilevel (1885-2014, 128x64, z256)
    {
        "model_name": "UKESM z256 Multi-Mesh Interpolated",
        "id": "z256_past_interpolated",
        "folder": [
            base_path_ukesm
            + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256_holes_filled/test_runs/wandb_test_0{run_number}_no_forecasts"
            for run_number in [1, 2, 3]
        ],
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256_holes_filled/plots/hist/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_holes_filled_forecasts/z256_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_1985_2014_equiangular_wp_conservative_40_chunks_holes_filled/fields.zarr",
        "train_time": ["15h 45m 26s", "16h 26m 31s", "17h 30m 7s"],
    },
]


FUTURE_MODELS_NG_era5 = [
    # Nextgems conservative multilevel (1990-2020, 128x64, z64)
    {
        "model_name": "NextGEMS z64 Multi-Mesh",
        "id": "z64_future",
        "folder": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03_2049",
        "folder_long": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03_2049_long",
        "mae": None,
        "rmse": None,
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative_forecasts/z64_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/plots/future/",
        "train_time": ["6h 55m 40s", "4h 13m 48s", "7h 52m 48s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z128)
    {
        "model_name": "NextGEMS z128 Multi-Mesh",
        "id": "z128_future",
        "folder": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_03_2049",
        "folder_long": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_03_2049_long",
        "mae": None,
        "rmse": None,
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative_forecasts/z128_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/plots/future/",
        "train_time": ["10h 43m 33s", "6h 17m 45s", "10h 42m 10s"],
    },
    # Nextgems conservative multilevel (1990-2020, 128x64, z256)
    {
        "model_name": "NextGEMS z256 Multi-Mesh",
        "id": "z256_future",
        "folder": None,
        "folder_long": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/test_runs/wandb_test_03_2049_long",
        "mae": [],
        "rmse": [],
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative_forecasts/z256_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/plots/future/",
        "train_time": ["16h 53m 16s", "10h 33m 14s", "18h 10m 39s"],
    },
    # Nextgems conservative hierarchical (1990-2020, 128x64, z64)
    {
        "model_name": "NextGEMS z64 Hierarchical",
        "id": "z64 hierarchical_future",
        "folder": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/test_runs/wandb_test_03_2049",
        "folder_long": base_path
        + f"nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/test_runs/wandb_test_03_2049_long",
        "mae": None,
        "rmse": None,
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative_forecasts/z64_graph_fm.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/plots/future/",
        "train_time": ["23h 3m 26s", "15h 23m 25s", "27h 28m 57s"],
    },
    # Nextgems Persistence conservative
    {
        "model_name": "Persistence",
        "id": "persistence_future",
        "folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_2049_persistence",
        "folder_long": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_2049_persistence_long",
        "mae": None,
        "rmse": None,
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative_forecasts/z64_persistence.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative/fields.zarr",
        "plot_folder": base_path
        + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/plots/future_persistence/",
        "train_time": None,
    },
    # ERA5 tested on NextGEMS data
    {
        "model_name": "ERA5 z64 Multi-Mesh",
        "id": "z64_era5_future",
        "folder": base_path
        + f"era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_03_2049",
        "folder_long": base_path
        + f"era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_03_2049_long",
        "mae": None,
        "rmse": None,
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative_forecasts/z64_graphcast_era5.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049_equiangular_with_poles_conservative/fields.zarr",
        "plot_folder": base_path + "era5_1990_2020_6h-128x64_z64/plots/future/",
        "train_time": ["6h 53m 50s", "4h 11m 9s", "7h 33m 57s"],
    },
]

FUTURE_MODELS_UKESM = [
    # UKESM conservative multilevel (1885-2014, 128x64, z64)
    {
        "model_name": "UKESM z64 Multi-Mesh",
        "id": "z64_future",
        "folder": base_path_ukesm
        + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03_future_no_forecasts",
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64/plots/future/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_forecasts/z64_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative/fields.zarr",
        "train_time": ["6h 11m 32s", "5h 8m 49s", "6h 9m 42s"],
    },
    # UKESM conservative multilevel (1885-2014, 128x64, z128)
    {
        "model_name": "UKESM z128 Multi-Mesh",
        "id": "z128_future",
        "folder": base_path_ukesm
        + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_03_future_no_forecasts",
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z128/plots/future/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_forecasts/z128_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative/fields.zarr",
        "train_time": ["13h 56m 1s", "5h 41m 56s", "9h 57m 34s"],
    },
    # UKESM conservative multilevel (1885-2014, 128x64, z256)
    {
        "model_name": "UKESM z256 Multi-Mesh",
        "id": "z256_future",
        "folder": base_path_ukesm
        + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256/test_runs/wandb_test_03_future_no_forecasts",
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256/plots/future/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_forecasts/z256_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative/fields.zarr",
        "train_time": ["15h 45m 26s", "16h 26m 31s", "17h 30m 7s"],
    },
    # UKESM conservative hierarchical (1885-2014, 128x64, z64)
    {
        "model_name": "UKESM z64 Hierarchical",
        "id": "z64 hierarchical_future",
        "folder": base_path_ukesm
        + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/test_runs/wandb_test_03_future_no_forecasts",
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical/plots/future/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_forecasts/z64_graph_fm.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative/fields.zarr",
        "train_time": ["25h 13m 59s", "17h 26m 43s", "26h 34m 13s"],
    },
    # UKESM Persistence conservative
    {
        "model_name": "Persistence",
        "id": "persistence_future",
        "folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_persistence_future",
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64/plots/future_persistence/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_forecasts/z64_persistence.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative/fields.zarr",
        "train_time": None,
    },
]

FUTURE_MODELS_UKESM_HOLES_FILLED = [
    # UKESM conservative multilevel (1885-2014, 128x64, z64)
    {
        "model_name": "UKESM z64 Multi-Mesh Interpolated",
        "id": "z64_future_interpolated",
        "folder": base_path_ukesm
        + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64_holes_filled/test_runs/wandb_test_03_future_no_forecasts",
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64_holes_filled/plots/future/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_holes_filled_forecasts/z64_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_holes_filled/fields.zarr",
        "train_time": ["6h 11m 32s", "5h 8m 49s", "6h 9m 42s"],
    },
    # UKESM conservative multilevel (1885-2014, 128x64, z256)
    {
        "model_name": "UKESM z256 Multi-Mesh Interpolated",
        "id": "z256_future_interpolated",
        "folder": base_path_ukesm
        + f"ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256_holes_filled/test_runs/wandb_test_03_future_no_forecasts",
        "mae": [],
        "rmse": [],
        "plot_folder": base_path_ukesm
        + "ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256_holes_filled/plots/future/",
        "forecasts": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_holes_filled_forecasts/z256_graphcast.zarr",
        "ground_truth": "/hkfs/work/workspace/scratch/xo8179-ukesm/global_2097_2100_equiangular_wp_conservative_holes_filled/fields.zarr",
        "train_time": ["15h 45m 26s", "16h 26m 31s", "17h 30m 7s"],
    },
]


MASK_VALUES_UKESM_TEMPERATURE = {
    "925": 211.8,
    "850": 215.3,
    "700": 202.3,
    "500": 212.9,
    "300": 197.9,
    "200": 185.1,
}


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
        1, 2, figsize=(10, 7), subplot_kw={"projection": proj}
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
    cbar = plt.colorbar(im, ax=axes[0], shrink=0.5, aspect=20, pad=0.02)
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
    show=False,
):

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 1, figsize=(5, 3.5), subplot_kw={"projection": proj}
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
    axes.add_feature(cfeature.COASTLINE, linewidth=0.8, color="grey")
    axes.add_feature(cfeature.BORDERS, linewidth=0.5, color="grey")
    axes.set_title(title_left)
    cbar = plt.colorbar(im, ax=axes, shrink=0.5, aspect=30, pad=0.02)
    cbar.set_label(cbar_label_left)

    plt.tight_layout()
    if show:
        plt.show()
    else:
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
    show=False,
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
        zonal_mean = data.mean(dim="longitude", skipna=True)
    else:
        zonal_mean = data

    # average over time if present
    if "time" in zonal_mean.dims:
        zonal_mean = zonal_mean.mean(dim="time", skipna=True)

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
        max_val = max(abs(np.nanmin(vals)), abs(np.nanmax(vals))) * 1.2
        ax.set_ylim(centered_around - max_val, centered_around + max_val)

    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            base_save_folder + f"{save_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
