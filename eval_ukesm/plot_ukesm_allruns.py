import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import re

sys.path.insert(
    1, "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/"
)
sys.path.insert(
    1, "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/evaluation"
)
import plot_utils as utils
from neural_lam.configs import get_constants

C = get_constants("ukesm")

models = utils.PAST_MODELS_UKESM  # + future_models
future_models = utils.FUTURE_MODELS_UKESM

for model in models:

    for run in [1, 2, 3]:
        try:
            model["mae"].append(
                pd.read_csv(
                    model["folder"][run - 1] + "/test_mae.csv", header=None
                )
            )
        except Exception:
            model["mae"].append(
                pd.read_csv(
                    model["folder"][run - 1] + "/files/test_mae.csv",
                    header=None,
                )
            )
        try:
            model["rmse"].append(
                pd.read_csv(
                    model["folder"][run - 1] + "/test_rmse.csv", header=None
                )
            )
        except Exception:
            model["rmse"].append(
                pd.read_csv(
                    model["folder"][run - 1] + "/files/test_rmse.csv",
                    header=None,
                )
            )
for model in future_models:
    try:
        model["mae"] = pd.read_csv(
            model["folder"] + "/test_mae.csv", header=None
        )
    except Exception:
        model["mae"] = pd.read_csv(
            model["folder"] + "/files/test_mae.csv", header=None
        )
    try:
        model["rmse"] = pd.read_csv(
            model["folder"] + "/test_rmse.csv", header=None
        )
    except Exception:
        model["rmse"] = pd.read_csv(
            model["folder"] + "/files/test_rmse.csv", header=None
        )


def daily_average(series):
    arr = np.array(series)
    return arr.reshape(-1, 4).mean(axis=1)


# Compute the mean MAE across all 83 variables (columns) for each time step (row)


all_plots = [
    ("Geopotential at 500hPa", "m^2/s^2", "geopot", C._get_var_index("z500")),
    ("Temperature at 850hPa", "K", "temp", C._get_var_index("t", 850)),
    (
        "Specific humidity at 700hPa",
        "kg/kg",
        "spec_hum",
        C._get_var_index("q", 700),
    ),
    ("T2m", "K", "t2m", C._get_var_index("t_surf")),
    (
        "U-component of wind at 850hPa",
        "m/s",
        "wind_u",
        C._get_var_index("u", 850),
    ),
    (
        "V-component of wind at 850hPa",
        "m/s",
        "wind_v",
        C._get_var_index("v", 850),
    ),
    ("Total Precipitation", "m", "tp", C._get_var_index("tp")),
]

plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/plots"
figname = "error_allmodels"

for run_number in [1, 2, 3]:
    # run_number = 1

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
                f"{title} - MAE daily avg. - Test period: 2010-2014"
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
                f"{title} - RMSE daily avg. - Test period: 2010-2014"
            )
            ax[1].set_xlabel("Days")
            ax[1].set_ylim(0, max_y)
            ax[1].set_ylabel(f"Daily average RMSE in {unit}")
            # ax[1].legend(loc='upper left')
            ax[1].grid(True)
        fig.tight_layout()
        fig.savefig(
            f"{plot_folder}/model_rollout_mae_rmse_per_run/{figname}_{save_name}_daily_avg_run_{run_number}.png",
            dpi=400,
        )
        plt.close()


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
        ax[0].set_title(f"{title} - MAE daily avg. - Test period: 2097-2100")
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
        ax[1].set_title(f"{title} - RMSE daily avg. - Test period: 2097-2100")
        ax[1].set_xlabel("Days")
        ax[1].set_ylim(0, max_y)
        ax[1].set_ylabel(f"Daily average RMSE in {unit}")
        # ax[1].legend(loc='upper left')
        ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/model_rollout_mae_rmse_per_run_future/{figname}_{save_name}_daily_avg.png",
        dpi=400,
    )
    plt.close()


for title, unit, save_name, idx in all_plots:
    # Plotting
    all_daily_mae = []
    all_daily_rmse = []

    for model in models:
        all_daily_mae.append(daily_average(model["mae"][2].iloc[:, idx]))
        all_daily_rmse.append(daily_average(model["rmse"][2].iloc[:, idx]))

    for model in future_models:
        all_daily_mae.append(daily_average(model["mae"].iloc[:, idx]))
        all_daily_rmse.append(daily_average(model["rmse"].iloc[:, idx]))

    max_mae = max(max(lst) for lst in (all_daily_mae)) * 1.1
    max_rmse = max(max(lst) for lst in (all_daily_rmse)) * 1.1
    max_y = max_mae

    # --- Plot ---

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    for model, mae in zip(models, all_daily_mae[0 : len(models)]):
        ax[0].plot(
            mae,
            label=model["model_name"] + f", run 3",
        )
        ax[0].set_title(f"{title} - MAE daily avg. - Test period: 2010-2014")
        ax[0].set_xlabel("Days")
        ax[0].set_ylim(0, max_y)
        ax[0].set_ylabel(f"Daily average MAE in {unit}")
        ax[0].legend(loc="upper left")
        ax[0].grid(True)

    for model, mae in zip(future_models, all_daily_mae[len(models) :]):
        ax[1].plot(
            mae,
            label=model["model_name"] + f"",
        )
        ax[1].set_title(f"{title} - MAE daily avg. - Test period: 2097-2100")
        ax[1].set_xlabel("Days")
        ax[1].set_ylim(0, max_y)
        ax[1].set_ylabel(f"Daily average MAE in {unit}")
        ax[1].legend(loc="upper left")
        ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/model_rollout_mae_per_run_hist_and_future/{figname}_{save_name}_daily_avg.png",
        dpi=400,
    )
    plt.close()

    max_y = max_rmse

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    for model, rmse in zip(models, all_daily_rmse[0 : len(models)]):
        ax[0].plot(
            rmse,
            label=model["model_name"] + f", run 3",
        )
        ax[0].set_title(f"{title} - RMSE daily avg. - Test period: 2010-2014")
        ax[0].set_xlabel("Days")
        ax[0].set_ylim(0, max_y)
        ax[0].set_ylabel(f"Daily average RMSE in {unit}")
        ax[0].legend(loc="upper left")
        ax[0].grid(True)

    for model, rmse in zip(future_models, all_daily_rmse[len(models) :]):
        ax[1].plot(
            rmse,
            label=model["model_name"] + f"",
        )
        ax[1].set_title(f"{title} - RMSE daily avg. - Test period: 2097-2100")
        ax[1].set_xlabel("Days")
        ax[1].set_ylim(0, max_y)
        ax[1].set_ylabel(f"Daily average RMSE in {unit}")
        ax[1].legend(loc="upper left")
        ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/model_rollout_rmse_per_run_hist_and_future/{figname}_{save_name}_daily_avg.png",
        dpi=400,
    )
    plt.close()
