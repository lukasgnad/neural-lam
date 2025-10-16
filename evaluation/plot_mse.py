import pandas as pd
import matplotlib.pyplot as plt

import neural_lam.constants as c


all_vars = c.PARAM_NAMES_SHORT


plot_folder = "plots/"

marker_persistence = 's'
marker_2000_2022 = 'x'
marker_1980_2022 = 'o'
color_persistence = "darkred"
color_1980_2022_1 = "deepskyblue"
color_1980_2022_2 = "royalblue"
color_1980_2022_3 = "darkblue"
color_2000_2022_1 = "lightgreen"
color_2000_2022_2 = "mediumseagreen"
color_2000_2022_3 = "darkgreen"
color_z128_1 = "peachpuff"
color_z128_2 = "sandybrown"
color_z128_3 = "peru"
folder_2000 = "era5_2000_2022/test_runs/wandb_era5_2000_2022_graphcast_test_"
folder_1980_128x64_64hidden = "era5_1980_2022_6h-128x64_z64/test_runs/wandb_era5_1980_2022_6h_128_64_z64_01_graphcast_test_"
folder_1980_128x64_128hidden = "era5_1980_2022_6h-128x64_z128/test_runs/wandb_era5_1980_2022_6h_128_64_01_graphcast_test_"
folder_persistence = "persistence_1980_2022/wandb_era5_2000_2022_persistence_test_4"

# Compute the mean MAE across all 83 variables (columns) for each time step (row)
graphcast_2000_1_avg_mae = pd.read_csv(folder_2000 + '1/files/test_mae.csv', header=None)
graphcast_2000_2_avg_mae = pd.read_csv(folder_2000 + '2/files/test_mae.csv', header=None)
graphcast_2000_3_avg_mae = pd.read_csv(folder_2000 + '3/files/test_mae.csv', header=None)
graphcast_1980_1_avg_mae = pd.read_csv(folder_1980_128x64_64hidden + '1/files/test_mae.csv', header=None)
graphcast_1980_2_avg_mae = pd.read_csv(folder_1980_128x64_64hidden + '2/files/test_mae.csv', header=None)
graphcast_1980_3_avg_mae = pd.read_csv(folder_1980_128x64_64hidden + '3/files/test_mae.csv', header=None)
graphcast_1980_128x64_128hidden_1_avg_mae = pd.read_csv(folder_1980_128x64_128hidden + '1/files/test_mae.csv', header=None)
graphcast_1980_128x64_128hidden_2_avg_mae = pd.read_csv(folder_1980_128x64_128hidden + '2/files/test_mae.csv', header=None)
graphcast_1980_128x64_128hidden_3_avg_mae = pd.read_csv(folder_1980_128x64_128hidden + '3/files/test_mae.csv', header=None)
persistence_avg_mae = pd.read_csv(folder_persistence + '/files/test_mae.csv', header=None)

# Compute the mean RMSE across all 83 variables (columns) for each time step (row)
graphcast_2000_1_avg_rmse = pd.read_csv(folder_2000 + '1/files/test_rmse.csv', header=None)
graphcast_2000_2_avg_rmse = pd.read_csv(folder_2000 + '2/files/test_rmse.csv', header=None)
graphcast_2000_3_avg_rmse = pd.read_csv(folder_2000 + '3/files/test_rmse.csv', header=None)
graphcast_1980_1_avg_rmse = pd.read_csv(folder_1980_128x64_64hidden + '1/files/test_rmse.csv', header=None)
graphcast_1980_2_avg_rmse = pd.read_csv(folder_1980_128x64_64hidden + '2/files/test_rmse.csv', header=None)
graphcast_1980_3_avg_rmse = pd.read_csv(folder_1980_128x64_64hidden + '3/files/test_rmse.csv', header=None)
graphcast_1980_128x64_128hidden_1_avg_rmse = pd.read_csv(folder_1980_128x64_128hidden + '1/files/test_rmse.csv', header=None)
graphcast_1980_128x64_128hidden_2_avg_rmse = pd.read_csv(folder_1980_128x64_128hidden + '2/files/test_rmse.csv', header=None)
graphcast_1980_128x64_128hidden_3_avg_rmse = pd.read_csv(folder_1980_128x64_128hidden + '3/files/test_rmse.csv', header=None)
persistence_avg_rmse = pd.read_csv(folder_persistence + '/files/test_rmse.csv', header=None)



idx_geopot_500 = all_vars.index("z500")
idx_temp_850 = all_vars.index("t850")
idx_spec_hum_700 = all_vars.index("q700")
idx_t2m = all_vars.index("2t")
idx_wind_u_850 = all_vars.index("u850")
idx_wind_v_850 = all_vars.index("v850")

assert idx_geopot_500 == 7
assert idx_temp_850 == 36
assert idx_spec_hum_700 == 22
assert idx_t2m == 78 #-5
assert idx_wind_u_850 == 49
assert idx_wind_v_850 == 62

all_plots = [("Geopotential at 500hPa", "m^2/s^2", "geopot", idx_geopot_500),
             ("Temperature at 850hPa", "K", "temp", idx_temp_850),
             ("Specific humidity at 700hPa", "kg/kg", "spec_hum", idx_spec_hum_700),
             ("T2m", "K", "t2m", idx_t2m),
             ("U-component of wind at 850hPa", "m/s", "wind_u", idx_wind_u_850),
             ("V-component of wind at 850hPa", "m/s", "wind_v", idx_wind_v_850)]


for (title, unit, save_name, idx) in all_plots:
    # Plotting
    
    mae_128_1, mae_128_2, mae_128_3 = graphcast_1980_128x64_128hidden_1_avg_mae.iloc[:, idx], graphcast_1980_128x64_128hidden_2_avg_mae.iloc[:, idx], graphcast_1980_128x64_128hidden_3_avg_mae.iloc[:, idx]
    mae_64_1, mae_64_2, mae_64_3 = graphcast_1980_1_avg_mae.iloc[:, idx], graphcast_1980_2_avg_mae.iloc[:, idx], graphcast_1980_3_avg_mae.iloc[:, idx]
    
    rmse_128_1, rmse_128_2, rmse_128_3 = graphcast_1980_128x64_128hidden_1_avg_rmse.iloc[:, idx], graphcast_1980_128x64_128hidden_2_avg_rmse.iloc[:, idx], graphcast_1980_128x64_128hidden_3_avg_rmse.iloc[:, idx]
    rmse_64_1, rmse_64_2, rmse_64_3 = graphcast_1980_1_avg_rmse.iloc[:, idx], graphcast_1980_2_avg_rmse.iloc[:, idx], graphcast_1980_3_avg_rmse.iloc[:, idx]
    
    mae_2000_1, mae_2000_2, mae_2000_3 = graphcast_2000_1_avg_mae.iloc[:, idx], graphcast_2000_2_avg_mae.iloc[:, idx], graphcast_2000_3_avg_mae.iloc[:, idx]
    rmse_2000_1, rmse_2000_2, rmse_2000_3 = graphcast_2000_1_avg_rmse.iloc[:, idx], graphcast_2000_2_avg_rmse.iloc[:, idx], graphcast_2000_3_avg_rmse.iloc[:, idx]
    
    mae_pers = persistence_avg_mae.iloc[:, idx]
    rmse_pers = persistence_avg_rmse.iloc[:, idx]
    all_lists = [mae_128_1,mae_128_2,mae_128_3,mae_64_1,mae_64_2,mae_64_3,
                rmse_128_1,rmse_128_2,rmse_128_3,rmse_64_1,rmse_64_2,rmse_64_3,
                mae_pers, mae_2000_1,mae_2000_2,mae_2000_3, rmse_2000_1, rmse_2000_2, rmse_2000_3, rmse_pers]
    max_y = max(max(lst) for lst in all_lists) * 1.1
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(1,2,figsize=(20, 6))
    ax[0].plot(mae_128_1, label='GC_1 (1980-2018, 2.8125°, z=128)', marker=marker_2000_2022, color=color_z128_1)
    ax[0].plot(mae_128_2, label='GC_2 (1980-2018, 2.8125°, z=128)', marker=marker_2000_2022, color=color_z128_2)
    ax[0].plot(mae_128_3, label='GC_3 (1980-2018, 2.8125°, z=128)', marker=marker_2000_2022, color=color_z128_3)
    ax[0].plot(mae_64_1, label='GC_1 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_1)
    ax[0].plot(mae_64_2, label='GC_2 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_2)
    ax[0].plot(mae_64_3, label='GC_3 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_3)
    ax[0].plot(mae_pers, label='Persistence', marker=marker_persistence, color=color_persistence)
    ax[0].set_title(f"{title} - MAE (Test period: 2021.01.01 - 2022.12.31)")
    ax[0].set_xlabel('Forecast Step (6h)')
    ax[0].set_ylim(0, max_y)
    ax[0].set_ylabel(f'Average MAE in {unit}')
    ax[0].legend(loc='upper left')
    ax[0].grid(True)
    ax[1].plot(rmse_128_1, label='GC_1 (1980-2018, 2.8125°, z=128)', marker=marker_2000_2022, color=color_z128_1)
    ax[1].plot(rmse_128_2, label='GC_2 (1980-2018, 2.8125°, z=128)', marker=marker_2000_2022, color=color_z128_2)
    ax[1].plot(rmse_128_3, label='GC_3 (1980-2018, 2.8125°, z=128)', marker=marker_2000_2022, color=color_z128_3)
    ax[1].plot(rmse_64_1, label='GC_1 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_1)
    ax[1].plot(rmse_64_2, label='GC_2 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_2)
    ax[1].plot(rmse_64_3, label='GC_3 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_3)
    ax[1].plot(rmse_pers, label='Persistence', marker=marker_persistence, color=color_persistence)
    ax[1].set_title(f"{title} - RMSE (Test period: 2021.01.01 - 2022.12.31)")
    ax[1].set_xlabel('Forecast Step')
    ax[1].set_ylim(0, max_y)
    ax[1].set_ylabel(f'Average RMSE in {unit}')
    ax[1].legend(loc='upper left')
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(f'{plot_folder}{save_name}_error_z64_vs_z128.png')
    
    fig, ax = plt.subplots(1,2,figsize=(20, 6))
    ax[0].plot(mae_2000_1, label='GC_1 (2000-2018, 3°, z=64)', marker=marker_2000_2022, color=color_2000_2022_1)
    ax[0].plot(mae_2000_2, label='GC_2 (2000-2018, 3°, z=64)', marker=marker_2000_2022, color=color_2000_2022_2)
    ax[0].plot(mae_2000_3, label='GC_3 (2000-2018, 3°, z=64)', marker=marker_2000_2022, color=color_2000_2022_3)
    ax[0].plot(mae_64_1, label='GC_1 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_1)
    ax[0].plot(mae_64_2, label='GC_2 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_2)
    ax[0].plot(mae_64_3, label='GC_3 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_3)
    ax[0].plot(mae_pers, label='Persistence', marker=marker_persistence, color=color_persistence)
    ax[0].set_title(f"{title} - MAE (Test period: 2021.01.01 - 2022.12.31)")
    ax[0].set_xlabel('Forecast Step (6h)')
    ax[0].set_ylim(0, max_y)
    ax[0].set_ylabel(f'Average MAE in {unit}')
    ax[0].legend(loc='upper left')
    ax[0].grid(True)
    ax[1].plot(rmse_2000_1, label='GC_1 (2000-2018, 3°, z=64)', marker=marker_2000_2022, color=color_2000_2022_1)
    ax[1].plot(rmse_2000_2, label='GC_2 (2000-2018, 3°, z=64)', marker=marker_2000_2022, color=color_2000_2022_2)
    ax[1].plot(rmse_2000_3, label='GC_3 (2000-2018, 3°, z=64)', marker=marker_2000_2022, color=color_2000_2022_3)
    ax[1].plot(rmse_64_2, label='GC_2 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_2)
    ax[1].plot(rmse_64_3, label='GC_3 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_3)
    ax[1].plot(rmse_64_1, label='GC_1 (1980-2018, 2.8125°, z=64)', marker=marker_1980_2022, color=color_1980_2022_1)
    ax[1].plot(rmse_pers, label='Persistence', marker=marker_persistence, color=color_persistence)
    ax[1].set_title(f"{title} - RMSE (Test period: 2021.01.01 - 2022.12.31)")
    ax[1].set_xlabel('Forecast Step')
    ax[1].set_ylim(0, max_y)
    ax[1].set_ylabel(f'Average RMSE in {unit}')
    ax[1].legend(loc='upper left')
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(f'{plot_folder}{save_name}_error_2000_vs_1980.png')