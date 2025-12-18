import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import re
import plot_utils as util

sys.path.insert(
    1, "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam"
)

from neural_lam.configs import get_constants

plot_ukesm = False

c = get_constants("ukesm") if plot_ukesm else get_constants("nextgems")


def parse_time_to_hours(time_str):
    """Parse string like '6h 55m 40s' into float hours"""
    h, m, s = 0, 0, 0
    match = re.findall(r"(\d+)([hms])", time_str)
    for value, unit in match:
        value = int(value)
        if unit == "h":
            h = value
        elif unit == "m":
            m = value
        elif unit == "s":
            s = value
    return h + m / 60 + s / 3600


for run_number in [1, 2, 3]:
    all_vars = c.PARAM_NAMES_SHORT
    # run_number = 1

    plot_folder = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/evaluation/plots"

    # future_models = [
    #     # Nexgems mixed future
    #     {
    #         "model_name": "NextGEMS mixed (2049, 128x64, z64)",
    #         "folder": (
    #             base_path
    #             + "nextgems_1990_2020_6h-128x64_try2_z64/test_runs/wandb_test_03_2049_no_forecasts"
    #         ),
    #         "mae": None,
    #         "rmse": None,
    #     },
    #     # Nextgems conservative future
    #     {
    #         "model_name": "NextGEMS conservative weatherbench (2049, 128x64, z64)",
    #         "folder": (
    #             base_path
    #             + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/test_runs/wandb_test_03_2049_no_forecasts"
    #         ),
    #         "mae": None,
    #         "rmse": None,
    #     },
    #     # Nextgems conservative 128 future
    #     {
    #         "model_name": "NextGEMS conservative weatherbench (2049, 128x64, z128)",
    #         "folder": (
    #             base_path
    #             + "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/test_runs/wandb_test_03_2049_no_forecasts"
    #         ),
    #         "mae": None,
    #         "rmse": None,
    #     },
    # ]


import matplotlib.pyplot as plt
import numpy as np

models = (
    [x for x in util.PAST_MODELS_UKESM if x["train_time"] is not None]
    if plot_ukesm
    else [x for x in util.PAST_MODELS_NG_era5 if x["train_time"] is not None]
)

model_short_names = (
    [
        "UKESM z64-MM",
        "UKESM z128-MM",
        "UKESM z256-MM",
        "UKESM z64-H",
        "Persistence",
    ]
    if plot_ukesm
    else ["NG z64-MM", "NG z128-MM", "NG z256-MM", "NG z64-H", "Era5 z64-MM"]
)


# Example data (replace with your own)
model_times = []

for model in models:
    model_times.append([parse_time_to_hours(x) for x in model["train_time"]])

n_models = len(models)
n_steps = len(model_times[0])

# X positions for groups (steps)
x = np.arange(n_steps)

# Width of each bar
bar_width = 0.18

colors = (
    [f"C{i}" for i in range(len(models))]
    if plot_ukesm
    else [f"C{i}" for i in range(len(models) - 1)] + [f"C{len(models)}"]
)

print(colors)
# Plot
fig, ax = plt.subplots(figsize=(8, 5))

for train_step in [1, 2, 3]:
    for model_index, (train_time, model) in enumerate(zip(model_times, models)):

        if train_step == 1:
            ax.bar(
                ((train_step - 1) * (n_models + 1) + model_index) * bar_width,
                train_time[train_step - 1],
                width=bar_width,
                label=model_short_names[model_index],
                color=colors[model_index],
            )
        else:
            ax.bar(
                ((train_step - 1) * (n_models + 1) + model_index) * bar_width,
                train_time[train_step - 1],
                width=bar_width,
                color=colors[model_index],
            )

# Formatting
xtick_positions = [
    ((step - 1) * (n_models + 1) + (n_models - 1) / 2) * bar_width
    for step in [1, 2, 3]
]

ax.set_xticks(xtick_positions)
ax.set_xticklabels(
    [
        "Step 1 (70 epochs)",
        "Step 2 (20 epochs)",
        "Step 3 (20 epochs)",
    ]
)
ax.set_ylabel("Time in hours")
ax.grid()
ax.set_title(
    "Training Time for Models Trained on UKESM per Step"
    if plot_ukesm
    else "Training Time for Models Trained on NextGEMS or ERA5 per Step"
)
ax.legend(loc="upper center")

plt.tight_layout()
plt.savefig(
    (
        f"{plot_folder}/model_compute_times_UKESM.pdf"
        if plot_ukesm
        else f"{plot_folder}/model_compute_times_NG.pdf"
    ),
    dpi=400,
)
plt.savefig(
    (
        f"{plot_folder}/model_compute_times_UKESM.png"
        if plot_ukesm
        else f"{plot_folder}/model_compute_times_NG.png"
    ),
    dpi=800,
)
