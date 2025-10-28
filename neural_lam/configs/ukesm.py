# configs/ukesm.py
import numpy as np
from .base import BaseConfig


class UKESMConfig(BaseConfig):
    """Configuration for UKESM dataset."""

    VAL_STEP_LOG_ERRORS = np.array([1, 2, 5, 10, 20, 40])
    VAL_STEP_CHECKPOINTS = np.array([1, 20, 40])

    ATMOSPHERIC_PARAMS = [
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "relative_vorticity",
    ]
    ATMOSPHERIC_PARAMS_SHORT = ["q", "t", "u", "v", "w", "rvor"]
    ATMOSPHERIC_PARAMS_UNITS = ["kg/kg", "K", "m/s", "m/s", "Pa/s", "1/s"]

    PRESSURE_LEVELS = [925, 850, 700, 500, 300, 200]

    SURFACE_PARAMS = [
        "mean_sea_level_pressure",
        "geopotential_500",
        "total_precipitation_6hr",
        "surface_temperature",
        "surface_u_component_of_wind",
        "surface_v_component_of_wind",
    ]
    SURFACE_PARAMS_SHORT = ["msl", "z500", "tp", "t_surf", "u_surf", "v_surf"]
    SURFACE_PARAM_UNITS = ["Pa", "m²/s²", "kg/m²", "K", "m/s", "m/s"]

    PREPROCESSING_PARAMS = ["geopotential_at_surface", "land_sea_mask"]

    EVAL_PRESSURE_LEVELS = [200, 500, 850]

    GRID_SHAPE = (128, 64)
    GRID_LIMITS = [0.0, 357.1875, -90.0, 90.0]

    def __init__(self):
        super().__init__()

        assert self._get_var_index("z500") == 37
        assert self._get_var_index("q", 700) == 2
        assert self.GRID_STATE_DIM == 42

        self.VAR_LEADS_METRICS_WATCH = {
            self._get_var_index("z500"): [1, 10],
        }

        self.VAL_PLOT_VARS = {
            self._get_var_index("q", 700): np.array([2, 20]),
            self._get_var_index("t", 850): np.array([2, 20]),
        }

        # Needed in preprocessing -> create_parameter_weights()
        self.SURFACE_WEIGHT_LIST = [
            1.0 if var_name == "z500" or var_name == "t_surf" else 0.1
            for var_name in self.SURFACE_PARAMS_SHORT
        ]
