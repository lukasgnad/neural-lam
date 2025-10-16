import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.insert(
    1, "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam"
)

import neural_lam.constants as c

all_vars = c.PARAM_NAMES_SHORT


plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/evaluation/plots"


figname = "error_era5_nextgems_128_64"
base_path = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/"

past_models = [
    # Nextgems mixed
    {
        "model_name": "NextGEMS mixed (1990-2020, 128x64, z64)",
        "folder": (
            base_path
            + "nextgems_1990_2020_6h-128x64_try2_z64/test_runs/wandb_test_03"
        ),
        "mae": None,
        "rmse": None,
    },
    # Nextgems conservative
    {
        "model_name": "NextGEMS conservative weatherbench (1990-2020, 128x64, z64)",
        "folder": (
            base_path
            + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03"
        ),
        "mae": None,
        "rmse": None,
    },
    # Nextgems conservative 128
    {
        "model_name": "NextGEMS conservative weatherbench (1990-2020, 128x64, z128)",
        "folder": (
            base_path
            + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_03"
        ),
        "mae": None,
        "rmse": None,
    },
    # Nextgems Persistence mixed
    {
        "model_name": "Persistence (NextGEMS mixed)",
        "folder": (
            base_path
            + "nextgems_1990_2020_6h-128x64_try2_z64/test_runs/wandb_test_persistence"
        ),
        "mae": None,
        "rmse": None,
    },
    # Nextgems Persistence conservative
    {
        "model_name": "Persistence (NextGEMS conservative weatherbench)",
        "folder": (
            base_path
            + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_persistence"
        ),
        "mae": None,
        "rmse": None,
    },
    # # Nextgems conservative future
    # {'model_name': "NextGEMS conservative (2049, 128x64, z64)",
    # 'folder': (
    #     base_path
    #     + "nextgems_1990_2020_6h-128x64_try2_conservative_z64/test_runs/wandb_test_03_2049"
    # ),
    # 'mae': None,
    # 'rmse': None,},
]

future_models = [
    # Nexgems mixed future
    {
        "model_name": "NextGEMS mixed (2049, 128x64, z64)",
        "folder": (
            base_path
            + "nextgems_1990_2020_6h-128x64_try2_z64/test_runs/wandb_test_03_2049_no_forecasts"
        ),
        "mae": None,
        "rmse": None,
    },
    # Nextgems conservative future
    {
        "model_name": "NextGEMS conservative weatherbench (2049, 128x64, z64)",
        "folder": (
            base_path
            + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03_2049_no_forecasts"
        ),
        "mae": None,
        "rmse": None,
    },
    # Nextgems conservative 128 future
    {
        "model_name": "NextGEMS conservative weatherbench (2049, 128x64, z128)",
        "folder": (
            base_path
            + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_03_2049_no_forecasts"
        ),
        "mae": None,
        "rmse": None,
    },
]


def daily_average(series):
    arr = np.array(series)
    return arr.reshape(-1, 4).mean(axis=1)


# Compute the mean MAE across all 83 variables (columns) for each time step (row)
models = past_models + future_models

for model in models:
    model["mae"] = pd.read_csv(
        model["folder"] + "/files/test_mae.csv", header=None
    )
    model["rmse"] = pd.read_csv(
        model["folder"] + "/files/test_rmse.csv", header=None
    )


idx_geopot_500 = all_vars.index("z500")
idx_temp_850 = all_vars.index("t850")
idx_spec_hum_700 = all_vars.index("q700")
idx_t2m = all_vars.index("2t")
idx_wind_u_850 = all_vars.index("u850")
idx_wind_v_850 = all_vars.index("v850")
idx_tp = all_vars.index("tp")

assert idx_geopot_500 == 7
assert idx_temp_850 == 36
assert idx_spec_hum_700 == 22
assert idx_t2m == 78  # -5
assert idx_wind_u_850 == 49
assert idx_wind_v_850 == 62
assert idx_tp == 82

all_plots = [
    ("Geopotential at 500hPa", "m^2/s^2", "geopot", idx_geopot_500),
    ("Temperature at 850hPa", "K", "temp", idx_temp_850),
    ("Specific humidity at 700hPa", "kg/kg", "spec_hum", idx_spec_hum_700),
    ("T2m", "K", "t2m", idx_t2m),
    ("U-component of wind at 850hPa", "m/s", "wind_u", idx_wind_u_850),
    ("V-component of wind at 850hPa", "m/s", "wind_v", idx_wind_v_850),
    ("Total Precipitation", "m", "tp", idx_tp),
]


for title, unit, save_name, idx in all_plots:
    # Plotting
    all_daily_mae = []
    all_daily_rmse = []
    for model in models:
        all_daily_mae.append(daily_average(model["mae"].iloc[:, idx]))
        all_daily_rmse.append(daily_average(model["rmse"].iloc[:, idx]))

    max_mae = max(max(lst) for lst in (all_daily_mae)) * 1.1
    max_rmse = max(max(lst) for lst in (all_daily_rmse)) * 1.1
    max_y = max(max_mae, max_rmse)

    # --- Plot ---

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    for model, mae in zip(models, all_daily_mae):
        ax[0].plot(
            mae,
            label=model["model_name"] + ", run 3",
        )
        ax[0].set_title(f"{title} - MAE daily avg. - Test period: 2018-2019")
        ax[0].set_xlabel("Days")
        ax[0].set_ylim(0, max_y)
        ax[0].set_ylabel(f"Daily average MAE in {unit}")
        ax[0].legend(loc="upper left")
        ax[0].grid(True)
    # ax[1].plot(rmse_x_1_daily, label=model_x_name + ', run 1', marker=marker_model_x, color=color_model_x_1)
    # ax[1].plot(rmse_x_2_daily, label=model_x_name + ', run 2', marker=marker_model_x, color=color_model_x_2)
    for model, rmse in zip(models, all_daily_rmse):
        ax[1].plot(
            rmse,
            label=model["model_name"] + ", run 3",
        )
        ax[1].set_title(f"{title} - RMSE daily avg. - Test period: 2018-2019")
        ax[1].set_xlabel("Days")
        ax[1].set_ylim(0, max_y)
        ax[1].set_ylabel(f"Daily average RMSE in {unit}")
        # ax[1].legend(loc='upper left')
        ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/model_rollout_mae_rmse/{figname}_{save_name}_daily_avg.png",
        dpi=400,
    )

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    for model, rmse in zip(past_models, all_daily_rmse[: len(past_models)]):
        ax[0].plot(
            rmse,
            label=model["model_name"] + ", run 3",
        )
        ax[0].set_title(f"{title} - RMSE daily avg. on test data (2018-2019)")
        ax[0].set_xlabel("Days")
        ax[0].set_ylim(0, max_rmse)
        ax[0].set_ylabel(f"Daily average RMSE in {unit}")
        ax[0].legend(loc="upper left")
        ax[0].grid(True)
    # ax[1].plot(rmse_x_1_daily, label=model_x_name + ', run 1', marker=marker_model_x, color=color_model_x_1)
    # ax[1].plot(rmse_x_2_daily, label=model_x_name + ', run 2', marker=marker_model_x, color=color_model_x_2)
    for model, rmse in zip(
        past_models,
        all_daily_rmse[
            len(past_models) : len(past_models) + len(future_models)
        ],
    ):
        ax[1].plot(
            rmse,
            label=model["model_name"] + ", run 3",
        )
        ax[1].set_title(f"{title} - RMSE daily avg. on future data (2049)")
        ax[1].set_xlabel("Days")
        ax[1].set_ylim(0, max_rmse)
        ax[1].set_ylabel(f"Daily average RMSE in {unit}")
        ax[1].legend(loc="upper left")
        ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/model_rollout_past_future_rmse/{figname}_{save_name}_past_vs_future_daily_avg.png",
        dpi=400,
    )
