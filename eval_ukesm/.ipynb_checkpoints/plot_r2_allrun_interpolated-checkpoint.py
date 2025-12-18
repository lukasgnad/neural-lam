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

# !!!
leave_out_64=True
# !!!



from neural_lam.configs import get_constants

c = get_constants("ukesm")

future_models_normal = utils.FUTURE_MODELS_UKESM
past_models_normal = utils.PAST_MODELS_UKESM

future_models_interpolated = utils.FUTURE_MODELS_UKESM_HOLES_FILLED
past_models_interpolated = utils.PAST_MODELS_UKESM_HOLES_FILLED

if leave_out_64:
    future_models = [future_models_normal[2]] + future_models_interpolated + [future_models_normal[4]]
    past_models = [past_models_normal[2]] + past_models_interpolated + [past_models_normal[4]]
else:
    future_models = [future_models_normal[0], future_models_normal[2]] + future_models_interpolated + [future_models_normal[4]]
    past_models = [past_models_normal[0], past_models_normal[2]] + past_models_interpolated + [past_models_normal[4]]


for idx, model in enumerate(future_models):
    model["r2"] = pd.read_csv(
        f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/rmse_r2_final/values_interpolated/r2_all_variables_{model['id']}.csv",
        index_col=0,
    )
    model["rmse"] = pd.read_csv(
        f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/rmse_r2_final/values_interpolated/rmse_all_variables_{model['id']}.csv",
        index_col=0,
    )
for idx, model in enumerate(past_models):
    model["r2"] = pd.read_csv(
        f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/rmse_r2_final/values_interpolated/r2_all_variables_{model['id']}.csv",
        index_col=0,
    )
    model["rmse"] = pd.read_csv(
        f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/rmse_r2_final/values_interpolated/rmse_all_variables_{model['id']}.csv",
        index_col=0,
    )

all_vars = utils.KEY_VARIABLES_UKESM

# Compute the mean MAE across all 83 variables (columns) for each time step (row)


plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/plots_interpolated"


figname = "r2_allmodels_future"



import matplotlib.lines as mlines

for display_name, unit, variable, plevel, _ in all_vars:

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    for i,model in enumerate(past_models):
        data = model["r2"].loc[variable]
        axes.plot(
            [x for x in range(1, 11)],
            data.groupby(np.arange(len(data)) // 4).mean(),
            label=model["model_name"] + f"",
            color=f'C{i+1}' if leave_out_64 else f'C{i}'
        )

    legend1 = axes.legend(title="Models", loc="lower left")
    axes.add_artist(legend1)
    for idx, model in enumerate(future_models):
        data = model["r2"].loc[variable]
        axes.plot(
            [x for x in range(1, 11)],
            data.groupby(np.arange(len(data)) // 4).mean(),
            label=model["model_name"] + f" Future",
            linestyle="dashed",
            color=f"C{idx+1}" if leave_out_64 else f"C{idx}",
        )

    # Add line style legend for past vs future
    past_line = mlines.Line2D([], [], color="black", linestyle="-", label="U_T1")
    future_line = mlines.Line2D(
        [], [], color="black", linestyle="--", label="U_T2"
    )
    axes.legend(
        handles=[past_line, future_line], loc="upper right", title="Test Set"
    )

    axes.set_title(f"{display_name} - R² Daily Average")
    axes.set_xlabel("Forecasting Time [Days]")
    axes.set_ylabel(f"R² score")
    axes.grid(True)
    fig.tight_layout()
    # fig.savefig(
    #     f"{plot_folder}/r2_plots/daily_together/daily_average_{figname}_{variable}_hist+future_together_z64_missing.png" if leave_out_64 else f"{plot_folder}/r2_plots/daily_together/daily_average_{figname}_{variable}_hist+future_together.png",
    #     dpi=400,
    # )
    fig.savefig(
        f"{plot_folder}/r2_plots/daily_average_{figname}_{variable}_hist+future_together_z64_missing.pdf" if leave_out_64 else f"{plot_folder}/r2_plots/daily_average_{figname}_{variable}_hist+future_together.pdf",
        dpi=400,
    )
    plt.close()





plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/plots_interpolated"


figname = "rmse_allmodels_future"



import matplotlib.lines as mlines

for display_name, unit, variable, plevel, _ in all_vars:

    fig, axes = plt.subplots(1, 1, figsize=(6,4))
    for i,model in enumerate(past_models):
        data = model["rmse"].loc[variable]
        axes.plot(
            [x for x in range(1, 11)],
            data.groupby(np.arange(len(data)) // 4).mean(),
            label=model["model_name"] + f"",
            color=f'C{i+1}' if leave_out_64 else f'C{i}'
        )

    legend1 = axes.legend(
        title="Models",
        loc="lower right",
    )
    axes.add_artist(legend1)
    for idx, model in enumerate(future_models):
        data = model["rmse"].loc[variable]
        axes.plot(
            [x for x in range(1, 11)],
            data.groupby(np.arange(len(data)) // 4).mean(),
            label=model["model_name"] + f" Future",
            linestyle="dashed",
            color=f"C{idx+1}" if leave_out_64 else f"C{idx}",
        )

    # Add line style legend for past vs future
    past_line = mlines.Line2D(
        [], [], color="black", linestyle="-", label="U_T1"
    )
    future_line = mlines.Line2D(
        [], [], color="black", linestyle="--", label="U_T2"
    )
    axes.legend(
        handles=[past_line, future_line], loc="upper left", title="Test Set"
    )

    axes.set_title(f"{display_name} - RMSE Daily Average")
    axes.set_xlabel("Forecasting Time [Days]")
    axes.set_ylabel(f"RMSE")
    axes.grid(True)
    fig.tight_layout()
    # fig.savefig(
    #     f"{plot_folder}/rmse_plots/daily_together/daily_average_{figname}_{variable}_hist+future_together_z64_missing.png"  if leave_out_64 else f"{plot_folder}/rmse_plots/daily_together/daily_average_{figname}_{variable}_hist+future_together.png",
    #     dpi=400,
    # )
    fig.savefig(
        f"{plot_folder}/rmse_plots/daily_average_{figname}_{variable}_hist+future_together_z64_missing.pdf" if leave_out_64 else f"{plot_folder}/rmse_plots/daily_average_{figname}_{variable}_hist+future_together.pdf",
        dpi=400,
    )
    plt.close()