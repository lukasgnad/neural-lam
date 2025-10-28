# !!! IMPORTANT !!!
# Constants for UKESM data!!

# Third-party
import cartopy
import numpy as np

WANDB_PROJECT = "neural-lam"

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 5, 10, 20, 40])
# Also save checkpoints for minimum loss at these lead times
VAL_STEP_CHECKPOINTS = np.array([1, 20, 40])

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = [
    "val_spsk_ratio",
    "val_rmse",
]
# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {
    37: [1, 10],  # z500
}

# Plot forecasts for these variables at given lead times during validation step
# Format is a dictionary that maps from a variable index to a list of
# lead time steps
VAL_PLOT_VARS = {
    2: np.array([2, 20]),  # q700
    7: np.array([2, 20]),  # t850
}

# During validation, plot example samples of latent variable from prior and
# variational distribution
LATENT_SAMPLES_PLOT = 4  # Number of samples to plot

# Following table 2 in GC
# Keys to read from fields zarr
ATMOSPHERIC_PARAMS = [
    # "geopotential", -> not present in ukesm data
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "relative_vorticity",
]  # times 6 pressure levels = 36 params

SURFACE_PARAMS = [
    "mean_sea_level_pressure",
    "geopotential_500",
]  # = 2 params
# Total = 38 params
PREPROCESSING_PARAMS = [
    "geopotential_at_surface",
	"land_sea_mask"
]

# Variable names
ATMOSPHERIC_PARAMS_SHORT = [
    "q",
    "t",
    "u",
    "v",
    "w",
    "rvor",
]
SURFACE_PARAMS_SHORT = ["msl", "z500"]


PRESSURE_LEVELS = [925, 850, 700, 500, 300, 200] # 6 levels


PARAM_NAMES_SHORT = [
    f"{param}{level}"
    for param in ATMOSPHERIC_PARAMS_SHORT
    for level in PRESSURE_LEVELS
] + SURFACE_PARAMS_SHORT

ATMOSPHERIC_PARAMS_UNITS = [
    "kg/kg",
    "K",
    "m/s",
    "m/s",
    "Pa/s",
    "1/s"
]
PARAM_UNITS = [
    unit for unit in ATMOSPHERIC_PARAMS_UNITS for level in PRESSURE_LEVELS
] + ["Pa", "m²/s²"]

# What variables (index) to plot during evaluation

EVAL_PLOT_VARS = np.concatenate(
    [
        level_start_i
        + np.arange(0, len(ATMOSPHERIC_PARAMS)) * len(PRESSURE_LEVELS)
        for level_start_i in (
            PRESSURE_LEVELS.index(level) for level in (200, 500, 850)
        )
    ]
    + [np.arange(36, 368)]  # Surface
)

# Projection and grid -> for 1.5° grid
#GRID_SHAPE = (240, 121)  # (long, lat)
# for 3°:
GRID_SHAPE = (128, 64)  # (long, lat)
#GRID_SHAPE = (120, 60)  # (long, lat)


# Create projection
MAP_PROJ = cartopy.crs.Robinson()
# for 1.5° grid
'''GRID_LIMITS = [
    -0.75,
    359.25,
    -90,
    90,
]'''
# for 3°:
# GRID_LIMITS = [-178.5, 178.5, -88.5, 88.5]
GRID_LIMITS = [
    0.0,            # min longitude
    357.1875,       # max longitude (128 * 2.8125)
    -90.0,          # min latitude
    90.0            # max latitude
]

# Time step length (hours)
TIME_STEP_LENGTH = 6

# Data dimensions
GRID_ORIGINAL_FORCING_DIM = 5  # 5 features
GRID_FORCING_DIM = GRID_ORIGINAL_FORCING_DIM * 3
# 5 features for 3 time-step window
GRID_STATE_DIM = 6 * 6 + 2  # 38
