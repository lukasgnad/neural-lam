import xarray as xr
from matplotlib import pyplot as plt
import xarray as xr
from scipy.stats import wasserstein_distance
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import plot_utils as plot


def rmse(y, yhat, dim=("time")):
    return np.sqrt(((y - yhat) ** 2).mean(dim))


def wasserstein_1d(a, b):
    return wasserstein_distance(a, b)


def shift_per_gridcell(train, test):
    # Drop time coords (keep only values along "time")
    train_anon = train.copy().chunk(dict(time=-1))
    test_anon = test.copy().chunk(dict(time=-1))
    train_anon = train_anon.assign_coords(time=np.arange(train.sizes["time"]))
    test_anon = test_anon.assign_coords(time=np.arange(test.sizes["time"]))

    return xr.apply_ufunc(
        wasserstein_1d,
        train_anon,
        test_anon,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


base_save_folder = "plots/dist_shift_performance/"

base_path = "/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/%s/test_runs/wandb_test_%s"
base_path_2049 = "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049_equiangular_with_poles_conservative/forecasts/z64_checkpoint_03_last.zarr"

nextgems_1990_2020_equiangular_wp_cons_z64 = xr.open_zarr(
    "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative/forecasts/z64_checkpoint_03_last.zarr"
)
nextgems_2049_equiangular_wp_cons_z64 = xr.open_zarr(
    "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049_equiangular_with_poles_conservative/forecasts/z64_checkpoint_03_last.zarr",
    decode_timedelta=True,
)

nextgems_equiangular_wp_cons_truth_1819 = xr.open_zarr(
    "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative/fields.zarr"
).sel(time=slice("2018", "2019"))
nextgems_equiangular_wp_cons_truth_49 = xr.open_zarr(
    "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049_equiangular_with_poles_conservative/fields.zarr"
)

nextgems_1990_2020_mixed_z64 = xr.open_zarr(
    "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_try2/forecasts/z64_checkpoint_03_last.zarr"
)
nextgems_2049_mixed_z64 = xr.open_zarr(
    "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049/forecasts/z64_checkpoint_03_last.zarr",
    decode_timedelta=True,
)

nextgems_mixed_truth_1819 = xr.open_zarr(
    "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_1990_2020_6h-128x64_try2/fields.zarr"
).sel(time=slice("2018", "2019"))
nextgems_mixed_truth_49 = xr.open_zarr(
    "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2049/fields.zarr"
)


# mixed = True
historical_year = "2019"
future_year = "2049"
error_after_x_steps = 19

for mixed in [True, False]:
    for display_name, unit, variable, plevel in plot.KEY_VARIABLES:

        info_to_save = (
            ("mixed_" if mixed else "cons_")
            + variable
            + (f"_{plevel}hPa" if plevel else "")
            + f"_{error_after_x_steps}step_error"
        )

        if mixed:
            truth_1819 = nextgems_mixed_truth_1819.sel(
                time=historical_year
            ).isel(time=slice(0, -1))[variable]
            truth_49 = nextgems_mixed_truth_49.sel(time=future_year)[variable]
            forecast_1819 = nextgems_1990_2020_mixed_z64.sel(
                time=historical_year
            )[variable]
            forecast_49 = nextgems_2049_mixed_z64.sel(time=future_year)[
                variable
            ]
        else:
            truth_1819 = nextgems_equiangular_wp_cons_truth_1819.sel(
                time=historical_year
            )[variable]
            truth_49 = nextgems_equiangular_wp_cons_truth_49.sel(
                time=future_year
            )[variable]
            forecast_1819 = nextgems_1990_2020_equiangular_wp_cons_z64.sel(
                time=historical_year
            )[variable]
            forecast_49 = nextgems_2049_equiangular_wp_cons_z64.sel(
                time=future_year
            )[variable]

        if plevel:
            truth_1819 = truth_1819.sel(level=plevel)
            truth_49 = truth_49.sel(level=plevel)
            forecast_1819 = forecast_1819.sel(level=plevel)
            forecast_49 = forecast_49.sel(level=plevel)

        performance_1819 = rmse(
            truth_1819,
            forecast_1819.isel(prediction_timedelta=error_after_x_steps),
        )
        performance_2049 = rmse(
            truth_49, forecast_49.isel(prediction_timedelta=error_after_x_steps)
        )

        shift_map_2049 = shift_per_gridcell(truth_1819, truth_49)
        relative_performance_error = performance_2049 / performance_1819
        absolute_performance_error = performance_2049 - performance_1819
        plot.plot_double_plot(
            data_left=shift_map_2049.T,
            title_left=f"Distribution shift (Wasserstein, 2049 vs 2019)\n{display_name}",
            cbar_label_left="Distribution shift",
            vbounds_left=None,
            cmap_left="Reds",
            data_right=relative_performance_error.T,
            title_right=f"Relative performance degradation (RMSE 2049 / 2019)\n{display_name}",
            cbar_label_right="Relative performance degradation",
            vbounds_right=(0.3, 1.7),
            cmap_right="coolwarm",
            base_save_folder=base_save_folder,
            save_name=f"shift_relative_performance_{info_to_save}",
        )
        plot.plot_double_plot(
            data_left=relative_performance_error.T,
            title_left=f"Relative performance degradation (RMSE 2049 / 2019)\n{display_name}",
            cbar_label_left="Relative performance degradation",
            vbounds_left=(0.3, 1.7),
            cmap_left="coolwarm",
            data_right=absolute_performance_error.T,
            title_right=f"Absolute performance degradation (RMSE 2049 - 2019)\n{display_name}",
            cbar_label_right="Absolute performance degradation",
            vbounds_right=None,
            cmap_right="coolwarm",
            base_save_folder=base_save_folder,
            save_name=f"absolute_relative_performance_{info_to_save}",
        )
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            1, 1, figsize=(7, 6), subplot_kw={"projection": proj}
        )
        (relative_performance_error / shift_map_2049).T.plot(
            ax=axes,
            cmap="Reds",
            transform=proj,
            cbar_kwargs={
                "shrink": 0.5,
                "aspect": 25,
                "label": "Normalized difference",
            },
        )
        axes.add_feature(cfeature.COASTLINE, linewidth=0.8)
        axes.add_feature(cfeature.BORDERS, linewidth=0.5)
        axes.set_title(
            f"Relative performance degradation / Distribution shift (Wasserstein)\n2049 vs 2019, {display_name}"
        )

        plt.tight_layout()
        plt.savefig(
            base_save_folder
            + f"normalized_performance_degradation_{info_to_save}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Finished var {display_name}")
