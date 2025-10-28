# configs/base.py
import numpy as np
import cartopy.crs as ccrs


class BaseConfig:
    """Base configuration with shared attributes for all datasets."""

    WANDB_PROJECT = "neural-lam"

    # Common metrics
    METRICS_WATCH = ["val_spsk_ratio", "val_rmse"]

    # Default latent sample plots
    LATENT_SAMPLES_PLOT = 4

    # Default projection
    MAP_PROJ = ccrs.Robinson()

    # Common training constants
    TIME_STEP_LENGTH = 6

    # Forcing window
    GRID_ORIGINAL_FORCING_DIM = 5
    GRID_FORCING_DIM = GRID_ORIGINAL_FORCING_DIM * 3

    def _get_var_index(self, var_short, pressure_level=None):
        # Get 3D var index
        if pressure_level:
            return self.ATMOSPHERIC_PARAMS_SHORT.index(var_short) * len(
                self.PRESSURE_LEVELS
            ) + self.PRESSURE_LEVELS.index(pressure_level)
        # Get 2D var index
        else:
            return len(self.ATMOSPHERIC_PARAMS_SHORT) * len(
                self.PRESSURE_LEVELS
            ) + self.SURFACE_PARAMS_SHORT.index(var_short)

    def __init__(self):

        self.GRID_STATE_DIM = len(self.ATMOSPHERIC_PARAMS) * len(
            self.PRESSURE_LEVELS
        ) + len(self.SURFACE_PARAMS)

        self.EVAL_PLOT_VARS = np.concatenate(
            [
                level_start_i
                + np.arange(0, len(self.ATMOSPHERIC_PARAMS))
                * len(self.PRESSURE_LEVELS)
                for level_start_i in (
                    self.PRESSURE_LEVELS.index(level)
                    for level in self.EVAL_PRESSURE_LEVELS
                )
            ]
            + [
                np.arange(
                    len(self.ATMOSPHERIC_PARAMS) * len(self.PRESSURE_LEVELS),
                    self.GRID_STATE_DIM,
                )
            ]  # Surface vars
        )

        # Derived attributes
        self.PARAM_NAMES_SHORT = [
            f"{param}{level}"
            for param in self.ATMOSPHERIC_PARAMS_SHORT
            for level in self.PRESSURE_LEVELS
        ] + self.SURFACE_PARAMS_SHORT

        self.PARAM_UNITS = [
            unit
            for unit in self.ATMOSPHERIC_PARAMS_UNITS
            for _ in self.PRESSURE_LEVELS
        ] + self.SURFACE_PARAM_UNITS

    def summary(self):
        """Pretty summary for debugging or logging."""
        print(f"Dataset: {self.__class__.__name__}")
        print(f"Grid shape: {self.GRID_SHAPE}")
        print(f"Pressure levels: {self.PRESSURE_LEVELS}")
        print(f"Variables: {len(self.PARAM_NAMES_SHORT)} total")
