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

marker_persistence = "s"
marker_model_x = "x"
marker_model_y = "o"
marker_model_z = "."
marker_model_z2 = "."
color_persistence = "darkred"
color_persistence_2 = "sandybrown"
color_model_x_1 = "deepskyblue"
color_model_x_2 = "royalblue"
color_model_x_3 = "darkblue"
color_model_y_1 = "lightgreen"
color_model_y_2 = "mediumseagreen"
color_model_y_3 = "darkgreen"
# color_model_z_1 = "peachpuff"
# color_model_z_2 = "sandybrown"
color_model_z_3 = "fuchsia"
color_model_z2_3 = "peachpuff"

figname = "error_era5_nextgems_128_64"
base_path = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/"


# Nextgems
model_x_name = "NextGEMS mixed (1990-2020, 128x64, z64)"
folder_model_x = (
    base_path
    + "nextgems_1990_2020_6h-128x64_try2_z64/test_runs/wandb_test_0"
)

# Era5
# model_y_name = "ERA5 (1990-2020, 128x64, z64)"
# folder_model_y = base_path + "era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_test_"
model_y_name = "NextGEMS conservative weatherbench (1990-2020, 128x64, z64)"
folder_model_y = (
    base_path
    + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_0"
)

# Persistence
persistence_name = "Persistence (NextGEMS mixed)"
folder_persistence = (
    base_path
    + "nextgems_1990_2020_6h-128x64_try2_z64/test_runs/wandb_test_persistence"
)
# Persistence
persistence_2_name = "Persistence (NextGEMS conservative weatherbench)"
folder_persistence_2 = (
    base_path
    + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_persistence"
)


# Nextgems
model_z_name = "NextGEMS conservative weatherbench (2049, 128x64, z64)"
folder_model_z = (
    base_path
    + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03_2049_no_forecasts"
)

# Nextgems
model_z2_name = "NextGEMS mixed (2049, 128x64, z64)"
folder_model_z2 = (
    base_path
    + "nextgems_1990_2020_6h-128x64_try2_z64/test_runs/wandb_test_03_2049_no_forecasts"
)


# Compute the mean MAE across all 83 variables (columns) for each time step (row)
graphcast_model_x_1_avg_mae = pd.read_csv(
    folder_model_x + "1/files/test_mae.csv", header=None
)
graphcast_model_x_2_avg_mae = pd.read_csv(
    folder_model_x + "2/files/test_mae.csv", header=None
)
graphcast_model_x_3_avg_mae = pd.read_csv(
    folder_model_x + "3/files/test_mae.csv", header=None
)
graphcast_model_y_1_avg_mae = pd.read_csv(
    folder_model_y + "1/files/test_mae.csv", header=None
)
graphcast_model_y_2_avg_mae = pd.read_csv(
    folder_model_y + "2/files/test_mae.csv", header=None
)
graphcast_model_y_3_avg_mae = pd.read_csv(
    folder_model_y + "3/files/test_mae.csv", header=None
)
graphcast_model_z_3_avg_mae = pd.read_csv(
    folder_model_z + "/files/test_mae.csv", header=None
)
graphcast_model_z2_3_avg_mae = pd.read_csv(
    folder_model_z2 + "/files/test_mae.csv", header=None
)
persistence_avg_mae = pd.read_csv(
    folder_persistence + "/files/test_mae.csv", header=None
)
persistence_2_avg_mae = pd.read_csv(
    folder_persistence_2 + "/files/test_mae.csv", header=None
)

# Compute the mean RMSE across all 83 variables (columns) for each time step (row)
graphcast_model_x_1_avg_rmse = pd.read_csv(
    folder_model_x + "1/files/test_rmse.csv", header=None
)
graphcast_model_x_2_avg_rmse = pd.read_csv(
    folder_model_x + "2/files/test_rmse.csv", header=None
)
graphcast_model_x_3_avg_rmse = pd.read_csv(
    folder_model_x + "3/files/test_rmse.csv", header=None
)
graphcast_model_y_1_avg_rmse = pd.read_csv(
    folder_model_y + "1/files/test_rmse.csv", header=None
)
graphcast_model_y_2_avg_rmse = pd.read_csv(
    folder_model_y + "2/files/test_rmse.csv", header=None
)
graphcast_model_y_3_avg_rmse = pd.read_csv(
    folder_model_y + "3/files/test_rmse.csv", header=None
)
graphcast_model_z_3_avg_rmse = pd.read_csv(
    folder_model_z + "/files/test_rmse.csv", header=None
)
graphcast_model_z2_3_avg_rmse = pd.read_csv(
    folder_model_z2 + "/files/test_rmse.csv", header=None
)
persistence_avg_rmse = pd.read_csv(
    folder_persistence + "/files/test_rmse.csv", header=None
)
persistence_2_avg_rmse = pd.read_csv(
    folder_persistence_2 + "/files/test_rmse.csv", header=None
)


idx_geopot_500 = all_vars.index("z500")
idx_temp_850 = all_vars.index("t850")
idx_spec_hum_700 = all_vars.index("q700")
idx_t2m = all_vars.index("2t")
idx_wind_u_850 = all_vars.index("u850")
idx_wind_v_850 = all_vars.index("v850")

assert idx_geopot_500 == 7
assert idx_temp_850 == 36
assert idx_spec_hum_700 == 22
assert idx_t2m == 78  # -5
assert idx_wind_u_850 == 49
assert idx_wind_v_850 == 62

all_plots = [
    ("Geopotential at 500hPa", "m^2/s^2", "geopot", idx_geopot_500),
    ("Temperature at 850hPa", "K", "temp", idx_temp_850),
    ("Specific humidity at 700hPa", "kg/kg", "spec_hum", idx_spec_hum_700),
    ("T2m", "K", "t2m", idx_t2m),
    ("U-component of wind at 850hPa", "m/s", "wind_u", idx_wind_u_850),
    ("V-component of wind at 850hPa", "m/s", "wind_v", idx_wind_v_850),
]


for title, unit, save_name, idx in all_plots:
    # Plotting

    mae_x_1, mae_x_2, mae_x_3 = (
        graphcast_model_x_1_avg_mae.iloc[:, idx],
        graphcast_model_x_2_avg_mae.iloc[:, idx],
        graphcast_model_x_3_avg_mae.iloc[:, idx],
    )
    mae_y_1, mae_y_2, mae_y_3 = (
        graphcast_model_y_1_avg_mae.iloc[:, idx],
        graphcast_model_y_2_avg_mae.iloc[:, idx],
        graphcast_model_y_3_avg_mae.iloc[:, idx],
    )
    mae_z_3, mae_z2_3 = (
        graphcast_model_z_3_avg_mae.iloc[:, idx],
        graphcast_model_z2_3_avg_mae.iloc[:, idx],
    )

    rmse_x_1, rmse_x_2, rmse_x_3 = (
        graphcast_model_x_1_avg_rmse.iloc[:, idx],
        graphcast_model_x_2_avg_rmse.iloc[:, idx],
        graphcast_model_x_3_avg_rmse.iloc[:, idx],
    )
    rmse_y_1, rmse_y_2, rmse_y_3 = (
        graphcast_model_y_1_avg_rmse.iloc[:, idx],
        graphcast_model_y_2_avg_rmse.iloc[:, idx],
        graphcast_model_y_3_avg_rmse.iloc[:, idx],
    )
    rmse_z_3, rmse_z2_3 = (
        graphcast_model_z_3_avg_rmse.iloc[:, idx],
        graphcast_model_z2_3_avg_rmse.iloc[:, idx],
    )

    mae_pers = persistence_avg_mae.iloc[:, idx]
    rmse_pers = persistence_avg_rmse.iloc[:, idx]

    mae_pers_2 = persistence_2_avg_mae.iloc[:, idx]
    rmse_pers_2 = persistence_2_avg_rmse.iloc[:, idx]

    all_lists = [
        mae_x_1,
        mae_x_2,
        mae_x_3,
        mae_y_1,
        mae_y_2,
        mae_y_3,
        rmse_x_1,
        rmse_x_2,
        rmse_x_3,
        rmse_y_1,
        rmse_y_2,
        rmse_y_3,
        mae_pers,
        rmse_pers,
        mae_pers_2,
        rmse_pers_2,
    ]
    max_y = max(max(lst) for lst in all_lists) * 1.1
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].plot(
        mae_x_1,
        label=model_x_name + ", run 1",
        marker=marker_model_x,
        color=color_model_x_1,
    )
    ax[0].plot(
        mae_x_2,
        label=model_x_name + ", run 2",
        marker=marker_model_x,
        color=color_model_x_2,
    )
    ax[0].plot(
        mae_x_3,
        label=model_x_name + ", run 3",
        marker=marker_model_x,
        color=color_model_x_3,
    )
    ax[0].plot(
        mae_y_1,
        label=model_y_name + ", run 1",
        marker=marker_model_y,
        color=color_model_y_1,
    )
    ax[0].plot(
        mae_y_2,
        label=model_y_name + ", run 2",
        marker=marker_model_y,
        color=color_model_y_2,
    )
    ax[0].plot(
        mae_y_3,
        label=model_y_name + ", run 3",
        marker=marker_model_y,
        color=color_model_y_3,
    )
    ax[0].plot(
        mae_pers,
        label=persistence_name,
        marker=marker_persistence,
        color=color_persistence,
    )
    ax[0].plot(
        mae_pers_2,
        label=persistence_2_name,
        marker=marker_persistence,
        color=color_persistence_2,
    )
    ax[0].set_title(f"{title} - MAE - Test period: 2018-2019")
    ax[0].set_xlabel("Forecast Step (6h)")
    ax[0].set_ylim(0, max_y)
    ax[0].set_ylabel(f"Average MAE in {unit}")
    ax[0].legend(loc="upper left")
    ax[0].grid(True)
    ax[1].plot(
        rmse_x_1,
        label=model_x_name + ", run 1",
        marker=marker_model_x,
        color=color_model_x_1,
    )
    ax[1].plot(
        rmse_x_2,
        label=model_x_name + ", run 2",
        marker=marker_model_x,
        color=color_model_x_2,
    )
    ax[1].plot(
        rmse_x_3,
        label=model_x_name + ", run 3",
        marker=marker_model_x,
        color=color_model_x_3,
    )
    ax[1].plot(
        rmse_y_1,
        label=model_y_name + ", run 1",
        marker=marker_model_y,
        color=color_model_y_1,
    )
    ax[1].plot(
        rmse_y_2,
        label=model_y_name + ", run 2",
        marker=marker_model_y,
        color=color_model_y_2,
    )
    ax[1].plot(
        rmse_y_3,
        label=model_y_name + ", run 3",
        marker=marker_model_y,
        color=color_model_y_3,
    )
    ax[1].plot(
        rmse_pers,
        label=persistence_name,
        marker=marker_persistence,
        color=color_persistence,
    )
    ax[1].plot(
        rmse_pers_2,
        label=persistence_2_name,
        marker=marker_persistence,
        color=color_persistence_2,
    )
    ax[1].set_title(f"{title} - RMSE - Test period: 2018-2019")
    ax[1].set_xlabel("Forecast Step (6h)")
    ax[1].set_ylim(0, max_y)
    ax[1].set_ylabel(f"Average RMSE in {unit}")
    # ax[1].legend(loc='upper left')
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(f"{plot_folder}/{figname}_{save_name}.png", dpi=400)

    # --- Compute daily averages (assuming len % 4 == 0) ---
    def daily_average(series):
        arr = np.array(series)
        return arr.reshape(-1, 4).mean(axis=1)

    mae_x_1_daily, mae_x_2_daily, mae_x_3_daily = (
        daily_average(mae_x_1),
        daily_average(mae_x_2),
        daily_average(mae_x_3),
    )
    mae_y_1_daily, mae_y_2_daily, mae_y_3_daily = (
        daily_average(mae_y_1),
        daily_average(mae_y_2),
        daily_average(mae_y_3),
    )
    mae_z_3_daily, mae_z2_3_daily = daily_average(mae_z_3), daily_average(
        mae_z2_3
    )
    mae_pers_daily, mae_pers_2_daily = daily_average(mae_pers), daily_average(
        mae_pers_2
    )
    rmse_x_1_daily, rmse_x_2_daily, rmse_x_3_daily = (
        daily_average(rmse_x_1),
        daily_average(rmse_x_2),
        daily_average(rmse_x_3),
    )
    rmse_y_1_daily, rmse_y_2_daily, rmse_y_3_daily = (
        daily_average(rmse_y_1),
        daily_average(rmse_y_2),
        daily_average(rmse_y_3),
    )
    rmse_z_3_daily, rmse_z2_3_daily = daily_average(rmse_z_3), daily_average(
        rmse_z2_3
    )
    rmse_pers_daily, rmse_pers_2_daily = daily_average(
        rmse_pers
    ), daily_average(rmse_pers_2)

    # --- Determine y-axis max ---
    all_daily = [
        mae_x_3_daily,
        mae_y_3_daily,
        mae_pers_daily,
        mae_pers_2_daily,
        rmse_x_3_daily,
        rmse_y_3_daily,
        mae_z_3_daily,
        mae_z2_3_daily,
        rmse_z_3_daily,
        rmse_z2_3_daily,
        rmse_pers_daily,
        rmse_pers_2_daily,
    ]
    max_y = max(max(lst) for lst in all_daily) * 1.1

    # --- Plot ---

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    # ax[0].plot(mae_x_1_daily, label=model_x_name + ', run 1', marker=marker_model_x, color=color_model_x_1)
    # ax[0].plot(mae_x_2_daily, label=model_x_name + ', run 2', marker=marker_model_x, color=color_model_x_2)
    ax[0].plot(
        mae_x_3_daily,
        label=model_x_name + ", run 3",
        marker=marker_model_x,
        color=color_model_x_3,
    )
    # ax[0].plot(mae_y_1_daily, label=model_y_name + ', run 1', marker=marker_model_y, color=color_model_y_1)
    # ax[0].plot(mae_y_2_daily, label=model_y_name + ', run 2', marker=marker_model_y, color=color_model_y_2)
    ax[0].plot(
        mae_y_3_daily,
        label=model_y_name + ", run 3",
        marker=marker_model_y,
        color=color_model_y_3,
    )
    ax[0].plot(
        mae_z_3_daily,
        label=model_z_name + ", run 3",
        marker=marker_model_z,
        color=color_model_z_3,
    )
    ax[0].plot(
        mae_z2_3_daily,
        label=model_z2_name + ", run 3",
        marker=marker_model_z2,
        color=color_model_z2_3,
    )
    ax[0].plot(
        mae_pers_daily,
        label=persistence_name,
        marker=marker_persistence,
        color=color_persistence,
    )
    ax[0].plot(
        mae_pers_2_daily,
        label=persistence_2_name,
        marker=marker_persistence,
        color=color_persistence_2,
    )
    ax[0].set_title(f"{title} - MAE daily avg. - Test period: 2018-2019")
    ax[0].set_xlabel("Days")
    ax[0].set_ylim(0, max_y)
    ax[0].set_ylabel(f"Daily average MAE in {unit}")
    ax[0].legend(loc="upper left")
    ax[0].grid(True)
    # ax[1].plot(rmse_x_1_daily, label=model_x_name + ', run 1', marker=marker_model_x, color=color_model_x_1)
    # ax[1].plot(rmse_x_2_daily, label=model_x_name + ', run 2', marker=marker_model_x, color=color_model_x_2)
    ax[1].plot(
        rmse_x_3_daily,
        label=model_x_name + ", run 3",
        marker=marker_model_x,
        color=color_model_x_3,
    )
    # ax[1].plot(rmse_y_1_daily, label=model_y_name + ', run 1', marker=marker_model_y, color=color_model_y_1)
    # ax[1].plot(rmse_y_2_daily, label=model_y_name + ', run 2', marker=marker_model_y, color=color_model_y_2)
    ax[1].plot(
        rmse_y_3_daily,
        label=model_y_name + ", run 3",
        marker=marker_model_y,
        color=color_model_y_3,
    )
    ax[1].plot(
        rmse_z_3_daily,
        label=model_z_name + ", run 3",
        marker=marker_model_z,
        color=color_model_z_3,
    )
    ax[1].plot(
        rmse_z2_3_daily,
        label=model_z2_name + ", run 3",
        marker=marker_model_z2,
        color=color_model_z2_3,
    )
    ax[1].plot(
        rmse_pers_daily,
        label=persistence_name,
        marker=marker_persistence,
        color=color_persistence,
    )
    ax[1].plot(
        rmse_pers_2_daily,
        label=persistence_2_name,
        marker=marker_persistence,
        color=color_persistence_2,
    )
    ax[1].set_title(f"{title} - RMSE daily avg. - Test period: 2018-2019")
    ax[1].set_xlabel("Days")
    ax[1].set_ylim(0, max_y)
    ax[1].set_ylabel(f"Daily average RMSE in {unit}")
    # ax[1].legend(loc='upper left')
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(f"{plot_folder}/{figname}_{save_name}_daily_avg.png", dpi=400)
