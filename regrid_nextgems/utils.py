import numpy as np
import xarray as xr
import xesmf as xe
import var_dicts as d


def get_regridder(var, ds):
    print(f"Getting regridder for var: {var}")
    # 2. Define target grid
    target_lon = np.arange(0, 360, 2.8125)
    target_lat = np.linspace(-90, 90, 64)
    # 3. Ensure input dataset has lat/lon in 1D
    # Conservative regridding requires correct orientation and spacing
    target_grid = xr.Dataset(
        {"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)}
    )

    # 3. Create regridder (adjust variable as needed)
    if var in d.bilinear_vars:
        print(f"Using bilinear")
        regridder = regridder_bilinear = xe.Regridder(
            ds, target_grid, "bilinear", periodic=True
        )
    elif var in d.conservative_vars:
        print(f"Using conservative")
        regridder = regridder_conservative = xe.Regridder(
            ds, target_grid, "conservative", periodic=True
        )
    elif var in d.nearest_vars:
        print(f"Using nearest")
        regridder = regridder_nearest = xe.Regridder(
            ds, target_grid, "nearest_s2d", periodic=True
        )
    print("Regridding....")
    return regridder


def get_conservative_regridder(var, ds):

    print(f"Getting regridder for var: {var}")
    # 2. Define target grid
    target_lon = np.arange(0, 360, 2.8125)
    target_lat = np.linspace(-90, 90, 64)
    # 3. Ensure input dataset has lat/lon in 1D
    # Conservative regridding requires correct orientation and spacing
    target_grid = xr.Dataset(
        {"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)}
    )

    # 3. Create regridder (adjust variable as needed)
    if var in d.bilinear_vars:
        print(f"Using bilinear")
        regridder = regridder_bilinear = xe.Regridder(
            ds, target_grid, "bilinear", periodic=True
        )
    elif var in d.conservative_vars:
        print(f"Using conservative")
        regridder = regridder_conservative = xe.Regridder(
            ds, target_grid, "conservative", periodic=True
        )
    elif var in d.nearest_vars:
        print(f"Using nearest")
        regridder = regridder_nearest = xe.Regridder(
            ds, target_grid, "nearest_s2d", periodic=True
        )
    print("Regridding....")
    return regridder
