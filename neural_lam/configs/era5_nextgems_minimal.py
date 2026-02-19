# configs/nextgems.py
import numpy as np
from .base import BaseConfig


class ERA5_NextGEMSConfig(BaseConfig):
    """Configuration for NextGEMS dataset."""

    VAL_STEP_LOG_ERRORS = np.array([1, 2, 5, 10, 20, 40])
    VAL_STEP_CHECKPOINTS = np.array([1, 20, 40])

    ATMOSPHERIC_PARAMS = [
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ]
    ATMOSPHERIC_PARAMS_SHORT = ["q", "t", "u", "v", "w"]
    ATMOSPHERIC_PARAMS_UNITS = ["kg/kg", "K", "m/s", "m/s", "Pa/s"]

    PRESSURE_LEVELS = [925, 850, 700, 500, 300, 200]

    SURFACE_PARAMS = [
        "geopotential_500",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_precipitation_6hr",
    ]
    SURFACE_PARAMS_SHORT = ["z", "2t", "10u", "10v", "msl", "tp"]
    SURFACE_PARAM_UNITS = ["m²/s²", "K", "m/s", "m/s", "Pa", "m"]

    PREPROCESSING_PARAMS = ["geopotential_at_surface", "land_sea_mask"]

    EVAL_PRESSURE_LEVELS = [200, 500, 850]

    GRID_SHAPE = (128, 64)
    GRID_LIMITS = [0.0, 357.1875, -90.0, 90.0]

    def __init__(self):
        super().__init__()

        assert self._get_var_index("q", 500) == 3
        assert self._get_var_index("2t") == 31
        assert self.GRID_STATE_DIM == 36

        self.VAR_LEADS_METRICS_WATCH = {
            self._get_var_index("z", 500): [1, 10],
            self._get_var_index("2t"): [1, 10],
            self._get_var_index("10u"): [1, 10],
        }

        self.VAL_PLOT_VARS = {
            self._get_var_index("z", 500): np.array([2, 20]),  
            self._get_var_index("q", 700): np.array([2, 20]),  
            self._get_var_index("z", 850): np.array([2, 20]), 
            self._get_var_index("2t"): np.array([2, 20]),  
            self._get_var_index("10u"): np.array([2, 20]),
            self._get_var_index("10v"): np.array([2, 20]), 
            self._get_var_index("tp"): np.array([2, 20]), 
        }

        # Needed in preprocessing -> create_parameter_weights()
        self.SURFACE_WEIGHT_LIST = [
            1.0 if var_name == "2t" else 0.1
            for var_name in self.SURFACE_PARAMS_SHORT
        ]


# # Third-party
# import cartopy
# import numpy as np

# WANDB_PROJECT = "neural-lam"

# # Log prediction error for these lead times
# VAL_STEP_LOG_ERRORS = np.array([1, 2, 5, 10, 20, 40])
# # Also save checkpoints for minimum loss at these lead times
# VAL_STEP_CHECKPOINTS = np.array([1, 20, 40])

# # Log these metrics to wandb as scalar values for
# # specific variables and lead times
# # List of metrics to watch, including any prefix (e.g. val_rmse)
# METRICS_WATCH = [
#     "val_spsk_ratio",
#     "val_rmse",
# ]
# # Dict with variables and lead times to log watched metrics for
# # Format is a dictionary that maps from a variable index to
# # a list of lead time steps
# VAR_LEADS_METRICS_WATCH = {
#     7: [1, 10],  # z500
#     78: [1, 10],  # 2t
#     79: [1, 10],  # 10u
# }

# # Plot forecasts for these variables at given lead times during validation step
# # Format is a dictionary that maps from a variable index to a list of
# # lead time steps
# VAL_PLOT_VARS = {
#     7: np.array([2, 20]),  # z500
#     22: np.array([2, 20]),  # q700
#     36: np.array([2, 20]),  # t850
#     78: np.array([2, 20]),  # 2t
#     79: np.array([2, 20]),  # 10u
#     80: np.array([2, 20]),  # 10v
#     82: np.array([2, 20]),  # tp
# }

# # During validation, plot example samples of latent variable from prior and
# # variational distribution
# LATENT_SAMPLES_PLOT = 4  # Number of samples to plot

# # Following table 2 in GC
# # Keys to read from fields zarr
# ATMOSPHERIC_PARAMS = [
#     "geopotential",
#     "specific_humidity",
#     "temperature",
#     "u_component_of_wind",
#     "v_component_of_wind",
#     "vertical_velocity",
# ]  # times 13 pressure levels = 78 params

# SURFACE_PARAMS = [
#     "2m_temperature",
#     "10m_u_component_of_wind",
#     "10m_v_component_of_wind",
#     "mean_sea_level_pressure",
#     "total_precipitation_6hr",
# ]  # = 5 params
# # Total = 83 params
# PREPROCESSING_PARAMS = [
#     "geopotential_at_surface",
# 	"land_sea_mask"
# ]

# # Variable names
# ATMOSPHERIC_PARAMS_SHORT = [
#     "z",
#     "q",
#     "t",
#     "u",
#     "v",
#     "w",
# ]
# SURFACE_PARAMS_SHORT = ["2t", "10u", "10v", "msl", "tp"]


# PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000, ]  # 13 levels


# PARAM_NAMES_SHORT = [
#     f"{param}{level}"
#     for param in ATMOSPHERIC_PARAMS_SHORT
#     for level in PRESSURE_LEVELS
# ] + SURFACE_PARAMS_SHORT

# ATMOSPHERIC_PARAMS_UNITS = [
#     "m²/s²",
#     "kg/kg",
#     "K",
#     "m/s",
#     "m/s",
#     "Pa/s",
# ]
# PARAM_UNITS = [
#     unit for unit in ATMOSPHERIC_PARAMS_UNITS for level in PRESSURE_LEVELS
# ] + ["K", "m/s", "m/s", "Pa", "m"]

# # What variables (index) to plot during evaluation

# EVAL_PLOT_VARS = np.concatenate(
#     [
#         level_start_i
#         + np.arange(0, len(ATMOSPHERIC_PARAMS)) * len(PRESSURE_LEVELS)
#         for level_start_i in (
#             PRESSURE_LEVELS.index(level) for level in (200, 500, 850)
#         )
#     ]
#     + [np.arange(78, 83)]  # Surface
# )

# # Projection and grid -> for 1.5° grid
# #GRID_SHAPE = (240, 121)  # (long, lat)
# # for 3°:
# GRID_SHAPE = (128, 64)  # (long, lat)
# #GRID_SHAPE = (120, 60)  # (long, lat)


# # Create projection
# MAP_PROJ = cartopy.crs.Robinson()
# # for 1.5° grid
# '''GRID_LIMITS = [
#     -0.75,
#     359.25,
#     -90,
#     90,
# ]'''
# # for 3°:
# # GRID_LIMITS = [-178.5, 178.5, -88.5, 88.5]
# GRID_LIMITS = [
#     0.0,            # min longitude
#     357.1875,       # max longitude (128 * 2.8125)
#     -90.0,          # min latitude
#     90.0            # max latitude
# ]

# # Time step length (hours)
# TIME_STEP_LENGTH = 6

# # Data dimensions
# GRID_ORIGINAL_FORCING_DIM = 5  # 5 features
# GRID_FORCING_DIM = GRID_ORIGINAL_FORCING_DIM * 3
# # 5 features for 3 time-step window
# GRID_STATE_DIM = 6 * 13 + 5  # 83
