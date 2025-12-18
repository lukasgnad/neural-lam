import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import xarray as xr
from scipy.stats import wasserstein_distance
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys

sys.path.insert(
    1, "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/evaluation"
)
import plot_utils as utils
import numpy as np

sys.path.insert(
    1, "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam"
)

from neural_lam.configs import get_constants

c = get_constants("ukesm")

models_temp = utils.PAST_MODELS_UKESM  # + future_models
future_models = utils.FUTURE_MODELS_UKESM

past_models = utils.PAST_MODELS_UKESM


model_resolution = [
    "z64",
    "z128",
    "z256",
    "z64 hierarchical",
    "persistence",
]

for idx, model in enumerate(future_models):
    model["rmse"] = pd.read_csv(
        f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/values/rmse_all_variables_{model_resolution[idx]}_future.csv",
        index_col=0,
    )
for idx, model in enumerate(past_models):
    model["rmse"] = pd.read_csv(
        f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/values/rmse_all_variables_{model_resolution[idx]}_past.csv",
        index_col=0,
    )

all_vars = utils.KEY_VARIABLES_UKESM

# Compute the mean MAE across all 83 variables (columns) for each time step (row)


plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/plots"


figname = "rmse_allmodels_future"

for display_name, unit, variable, plevel in all_vars:

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for model in past_models:
        axes[0].plot(
            [x * 6 for x in range(1, 41)],
            model["rmse"].loc[variable],
            label=model["model_name"] + f"",
        )
        axes[0].set_title(f"{display_name} - RMSE - Test Period: 2011-2014")
        axes[0].set_xlabel("Hours")
        axes[0].set_ylabel(f"RMSE")
        axes[0].legend()
        axes[0].grid(True)
    for model in future_models:
        axes[1].plot(
            [x * 6 for x in range(1, 41)],
            model["rmse"].loc[variable],
            label=model["model_name"] + f"",
        )
        axes[1].set_title(f"{display_name} - RMSE - Test Period: 2097-2100")
        axes[1].set_xlabel("Hours")
        axes[1].set_ylabel(f"RMSE")
        axes[1].legend()
        axes[1].grid(True)
    utils.sync_axes(axes, sync_x=False)

    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/rmse_plots/{figname}_{variable}_hist+future.png",
        dpi=400,
    )
    plt.close()

for display_name, unit, variable, plevel in all_vars:

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for model in past_models:
        data = model["rmse"].loc[variable]
        axes[0].plot(
            [x for x in range(1, 11)],
            data.groupby(np.arange(len(data)) // 4).mean(),
            label=model["model_name"] + f"",
        )
        axes[0].set_title(
            f"{display_name} - RMSE Daily Average - Test Period: 2011-2014"
        )
        axes[0].set_xlabel("Days")
        axes[0].set_ylabel(f"RMSE")
        axes[0].legend()
        axes[0].grid(True)
    for model in future_models:
        data = model["rmse"].loc[variable]
        axes[1].plot(
            [x for x in range(1, 11)],
            data.groupby(np.arange(len(data)) // 4).mean(),
            label=model["model_name"] + f"",
        )
        axes[1].set_title(
            f"{display_name} - RMSE Daily Average - Test Period: 2097-2100"
        )
        axes[1].set_xlabel("Days")
        axes[1].set_ylabel(f"RMSE")
        axes[1].legend()
        axes[1].grid(True)

    utils.sync_axes(axes, sync_x=False)
    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/rmse_plots/daily_average_{figname}_{variable}_hist+future.png",
        dpi=400,
    )
    plt.close()


# figname = "r2_allmodels_future"

# for display_name, unit, variable, plevel in all_vars:

#     fig, axes = plt.subplots(1, 1, figsize=(10, 6))
#     for model in past_models:
#         axes.plot(
#             [x * 6 for x in range(1, 41)],
#             model["r2"].loc[variable],
#             label=model["model_name"] + f"",
#         )
#         axes.set_title(f"{display_name} - R² - Test Period: 2011-2014")
#         axes.set_xlabel("Hours")
#         axes.set_ylabel(f"R² score")
#         axes.legend()
#         axes.grid(True)

#     fig.tight_layout()
#     fig.savefig(
#         f"{plot_folder}/r2_plots/past/{figname}_{variable}.png",
#         dpi=400,
#     )
#     plt.close()

# for display_name, unit, variable, plevel in all_vars:

#     fig, axes = plt.subplots(1, 1, figsize=(10, 6))
#     for model in past_models:
#         data = model["r2"].loc[variable]
#         axes.plot(
#             [x for x in range(1, 11)],
#             data.groupby(np.arange(len(data)) // 4).mean(),
#             label=model["model_name"] + f"",
#         )
#         axes.set_title(
#             f"{display_name} - R² Daily Average - Test Period: 2011-2014"
#         )
#         axes.set_xlabel("Days")
#         axes.set_ylabel(f"R² score")
#         axes.legend()
#         axes.grid(True)

#     fig.tight_layout()
#     fig.savefig(
#         f"{plot_folder}/r2_plots/past/daily_average_{figname}_{variable}.png",
#         dpi=400,
#     )
#     plt.close()
