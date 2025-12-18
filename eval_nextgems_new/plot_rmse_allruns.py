import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import xarray as xr
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

c = get_constants("nextgems")

models_temp = utils.PAST_MODELS_NG_era5  # + future_models
future_models = utils.FUTURE_MODELS_NG_era5

past_models = utils.PAST_MODELS_NG_era5


for idx, model in enumerate(future_models):
    model["rmse"] = pd.read_csv(
        f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_nextgems_new/values/rmse_all_variables_{model['id']}.csv",
        index_col=0,
    )
for idx, model in enumerate(past_models):
    model["rmse"] = pd.read_csv(
        f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_nextgems_new/values/rmse_all_variables_{model['id']}.csv",
        index_col=0,
    )

all_vars = utils.KEY_VARIABLES

# Compute the mean MAE across all 83 variables (columns) for each time step (row)


plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_nextgems_new/plots"


figname = "rmse_allmodels_future"

# for display_name, unit, variable, plevel, _ in all_vars:

#     fig, axes = plt.subplots(1, 2, figsize=(20, 6))
#     for model in past_models:
#         axes[0].plot(
#             [x * 6 for x in range(1, 41)],
#             model["rmse"].loc[variable],
#             label=model["model_name"] + f"",
#         )
#         axes[0].set_title(f"{display_name} - RMSE - Test Period: 2016-2019")
#         axes[0].set_xlabel("Forecasting Time [Hours]")
#         axes[0].set_ylabel(f"RMSE")
#         axes[0].legend()
#         axes[0].grid(True)
#     for model in future_models:
#         axes[1].plot(
#             [x * 6 for x in range(1, 41)],
#             model["rmse"].loc[variable],
#             label=model["model_name"] + f"",
#         )
#         axes[1].set_title(f"{display_name} - RMSE - Test Period: 2046-2049")
#         axes[1].set_xlabel("Forecasting Time [Hours]")
#         axes[1].set_ylabel(f"RMSE")
#         axes[1].legend()
#         axes[1].grid(True)
#     utils.sync_axes(axes, sync_x=False)

#     fig.tight_layout()
#     fig.savefig(
#         f"{plot_folder}/rmse_plots/hourly/{figname}_{variable}_hist+future.png",
#         dpi=400,
#     )
#     fig.savefig(
#         f"{plot_folder}/rmse_plots/hourly/{figname}_{variable}_hist+future.pdf",
#         dpi=400,
#     )
#     plt.close()

# for display_name, unit, variable, plevel, _ in all_vars:

#     fig, axes = plt.subplots(1, 2, figsize=(20, 6))
#     for model in past_models:
#         data = model["rmse"].loc[variable]
#         axes[0].plot(
#             [x for x in range(1, 11)],
#             data.groupby(np.arange(len(data)) // 4).mean(),
#             label=model["model_name"] + f"",
#         )
#         axes[0].set_title(
#             f"{display_name} - RMSE Daily Average - Test Period: 2016-2019"
#         )
#         axes[0].set_xlabel("Forecasting Time [Days]")
#         axes[0].set_ylabel(f"RMSE")
#         axes[0].legend()
#         axes[0].grid(True)
#     for model in future_models:
#         data = model["rmse"].loc[variable]
#         axes[1].plot(
#             [x for x in range(1, 11)],
#             data.groupby(np.arange(len(data)) // 4).mean(),
#             label=model["model_name"] + f"",
#         )
#         axes[1].set_title(
#             f"{display_name} - RMSE Daily Average - Test Period: 2046-2049"
#         )
#         axes[1].set_xlabel("Forecasting Time [Days]")
#         axes[1].set_ylabel(f"RMSE")
#         axes[1].legend()
#         axes[1].grid(True)

#     utils.sync_axes(axes, sync_x=False)
#     fig.tight_layout()
#     fig.savefig(
#         f"{plot_folder}/rmse_plots/daily/daily_average_{figname}_{variable}_hist+future.png",
#         dpi=400,
#     )
#     fig.savefig(
#         f"{plot_folder}/rmse_plots/daily/daily_average_{figname}_{variable}_hist+future.pdf",
#         dpi=400,
#     )
#     plt.close()
plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_nextgems_new/plots"


figname = "rmse_allmodels_future"

import matplotlib.lines as mlines

for display_name, unit, variable, plevel, _ in all_vars:

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    for model in past_models:
        data = model["rmse"].loc[variable]
        axes.plot(
            [x for x in range(1, 11)],
            data.groupby(np.arange(len(data)) // 4).mean(),
            label=model["model_name"] + f"",
        )

    legend1 = axes.legend(title="Models", loc="lower right")
    axes.add_artist(legend1)
    for idx, model in enumerate(future_models):
        data = model["rmse"].loc[variable]
        axes.plot(
            [x for x in range(1, 11)],
            data.groupby(np.arange(len(data)) // 4).mean(),
            label=model["model_name"] + f" Future",
            linestyle="dashed",
            color=f"C{idx}",
        )

    # Add line style legend for past vs future
    past_line = mlines.Line2D([], [], color="black", linestyle="-", label="T1")
    future_line = mlines.Line2D(
        [], [], color="black", linestyle="--", label="T2"
    )
    axes.legend(
        handles=[past_line, future_line],
        loc="upper left",
        title="Test Set",
    )

    axes.set_title(f"{display_name} - RMSE Daily Average")
    axes.set_xlabel("Forecasting Time [Days]")
    axes.set_ylabel(f"RMSE")
    axes.grid(True)
    fig.tight_layout()
    fig.savefig(
        f"{plot_folder}/rmse_plots/daily_together/daily_average_{figname}_{variable}_hist+future_together.png",
        dpi=400,
    )
    fig.savefig(
        f"{plot_folder}/rmse_plots/daily_together/daily_average_{figname}_{variable}_hist+future_together.pdf",
        dpi=400,
    )
    plt.close()
