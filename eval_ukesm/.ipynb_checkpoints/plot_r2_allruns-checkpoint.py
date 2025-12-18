import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import re
import plot_utils as utils

sys.path.insert(
    1, "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam"
)

from neural_lam.configs import get_constants

c = get_constants("nextgems")

models_temp = utils.PAST_MODELS_NG_era5  # + future_models
future_models_temp = utils.FUTURE_MODELS_NG_era5

models = [x for x in models_temp if not "ERA5" in x["model_name"]]
future_models = [x for x in future_models_temp if not "ERA5" in x["model_name"]]

for model in models:

    for run in [1, 2, 3]:
        try:
            model["mae"].append(
                pd.read_csv(
                    model["folder_long"][run - 1] + "/test_mae.csv", header=None
                )
            )
        except Exception:
            model["mae"].append(
                pd.read_csv(
                    model["folder_long"][run - 1] + "/files/test_mae.csv",
                    header=None,
                )
            )
        try:
            model["rmse"].append(
                pd.read_csv(
                    model["folder_long"][run - 1] + "/test_rmse.csv",
                    header=None,
                )
            )
        except Exception:
            model["rmse"].append(
                pd.read_csv(
                    model["folder_long"][run - 1] + "/files/test_rmse.csv",
                    header=None,
                )
            )
for model in future_models:
    try:
        model["mae"] = pd.read_csv(
            model["folder_long"] + "/test_mae.csv", header=None
        )
    except Exception:
        model["mae"] = pd.read_csv(
            model["folder_long"] + "/files/test_mae.csv", header=None
        )
    try:
        model["rmse"] = pd.read_csv(
            model["folder_long"] + "/test_rmse.csv", header=None
        )
    except Exception:
        model["rmse"] = pd.read_csv(
            model["folder_long"] + "/files/test_rmse.csv", header=None
        )


all_vars = c.PARAM_NAMES_SHORT
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


def daily_average(series):
    arr = np.array(series)
    return arr.reshape(-1, 4).mean(axis=1)


# Compute the mean MAE across all 83 variables (columns) for each time step (row)


all_plots = [
    ("Geopotential at 500hPa", "m^2/s^2", "geopot", idx_geopot_500),
    ("Temperature at 850hPa", "K", "temp", idx_temp_850),
    ("Specific humidity at 700hPa", "kg/kg", "spec_hum", idx_spec_hum_700),
    ("T2m", "K", "t2m", idx_t2m),
    ("U-component of wind at 850hPa", "m/s", "wind_u", idx_wind_u_850),
    ("V-component of wind at 850hPa", "m/s", "wind_v", idx_wind_v_850),
    ("Total Precipitation", "m", "tp", idx_tp),
]

for run_number in [1, 2, 3]:
    # run_number = 1

    plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/evaluation/plots"

    figname = "error_allmodels_long"
    base_path = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/"

    for title, unit, save_name, idx in all_plots:
        # Plotting
        all_daily_mae = []
        all_daily_rmse = []
        for model in models:
            all_daily_mae.append(
                daily_average(model["mae"][run_number - 1].iloc[:, idx])
            )
            all_daily_rmse.append(
                daily_average(model["rmse"][run_number - 1].iloc[:, idx])
            )

        max_mae = max(max(lst) for lst in (all_daily_mae)) * 1.1
        max_rmse = max(max(lst) for lst in (all_daily_rmse)) * 1.1
        max_y = max(max_mae, max_rmse)

        # --- Plot ---

        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        for model, mae in zip(models, all_daily_mae):
            ax[0].plot(
                mae,
                label=model["model_name"] + f", run {run_number}",
            )
            ax[0].set_title(
                f"{title} - MAE daily avg. - Test period: 2016-2019"
            )
            ax[0].set_xlabel("Days")
            ax[0].set_ylim(0, max_y)
            ax[0].set_ylabel(f"Daily average MAE in {unit}")
            ax[0].legend(loc="upper left")
            ax[0].grid(True)

        for model, rmse in zip(models, all_daily_rmse):
            ax[1].plot(
                rmse,
                label=model["model_name"] + f", run {run_number}",
            )
            ax[1].set_title(
                f"{title} - RMSE daily avg. - Test period: 2016-2019"
            )
            ax[1].set_xlabel("Days")
            ax[1].set_ylim(0, max_y)
            ax[1].set_ylabel(f"Daily average RMSE in {unit}")
            # ax[1].legend(loc='upper left')
            ax[1].grid(True)
        fig.tight_layout()
        fig.savefig(
            f"{plot_folder}/model_rollout_mae_rmse_per_run_long/{figname}_{save_name}_daily_avg_run_{run_number}.png",
            dpi=400,
        )
        plt.close()


plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/evaluation/plots"

figname = "error_allmodels_long"
base_path = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/"

for title, unit, save_name, idx in all_plots:
    # Plotting
    all_daily_mae = []
    all_daily_rmse = []
    for model in future_models:
        all_daily_mae.append(daily_average(model["mae"].iloc[:, idx]))
        all_daily_rmse.append(daily_average(model["rmse"].iloc[:, idx]))

    max_mae = max(max(lst) for lst in (all_daily_mae)) * 1.1
    max_rmse = max(max(lst) for lst in (all_daily_rmse)) * 1.1
    max_y = max(max_mae, max_rmse)

    # --- Plot ---

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    for model, mae in zip(future_models, all_daily_mae):
        ax[0].plot(
            mae,
            label=model["model_name"] + f"",
        )
        ax[0].set_title(f"{title} - MAE daily avg. - Test period: 2046-2049")
        ax[0].set_xlabel("Days")
        ax[0].set_ylim(0, max_y)
        ax[0].set_ylabel(f"Daily average MAE in {unit}")
        ax[0].legend(loc="upper left")
        ax[0].grid(True)

    for model, rmse in zip(future_models, all_daily_rmse):
        ax[1].plot(
            rmse,
            label=model["model_name"] + f"",
        )
        ax[1].set_title(f"{title} - RMSE daily avg. - Test period: 2046-2049")
        ax[1].set_xlabel("Days")
        ax[1].set_ylim(0, max_y)
        ax[1].set_ylabel(f"Daily average RMSE in {unit}")
        # ax[1].legend(loc='upper left')
        ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/model_rollout_mae_rmse_per_run_future_long/{figname}_{save_name}_daily_avg.png",
        dpi=400,
    )
    plt.close()
