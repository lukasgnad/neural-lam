import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

base = "/hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextGEMS_1990_2020_6h-128x64_transposed/"

ds = xr.open_dataset(base+"fields.zarr", )

def plot_num_nans(dataset):
    for var in dataset.variables:
        total_nans = 0
        if "time" in dataset[var].dims:
            for i in range(1990, 2020):
                total_nans = total_nans + np.isnan(dataset[var].sel(time=f'{i}').values.flatten()).sum()
            print(f"Number of NaNs for {var}: {total_nans}")
        else:
            print(f"Number of NaNs for {var}: {np.isnan(dataset[var].values.flatten()).sum()}")
        
#plot_num_nans(ds)
import pandas as pd
var = "geopotential"
nan_locations = []

for year in range(1990, 2020):
    da = ds[var].sel(time=str(year))
    mask = np.isnan(da)

    if mask.any():
        # Get indexes of NaNs
        idxs = np.argwhere(mask.values)
        for idx in idxs:
            # Extract coordinates for each NaN
            coords = {}
            for axis, dim in enumerate(da.dims):
                coord_val = da[dim].values[idx[axis]]
                coords[dim] = idx
            coords["year"] = year
            nan_locations.append(coords)

# Convert to DataFrame for easy viewing
df_nan_locations = pd.DataFrame(nan_locations)
print(df_nan_locations)