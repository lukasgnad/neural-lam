# Preprocessing for Training / Testing

The Preprocessing for training the model / testing the model comes in 4 steps:
- Create a graph mapping, this is done via create_global_mesh.py, and only needs to know the dataset dimensions.
- Create global grid features, utilizing 'geopotential_at_surface' and 'land_sea_mask' from era5. Done with  create_global_grid_features.py
- Calculate a global forcing, including time of year and time of day, as well as total solar incident radiation (TSI). Done with create_global_forcing.py, or better create_global_forcing_stable.py, which accounts for errors when dealing with 360 day years (i.e. UKESM).
- Calculate the mean and std per parameter, and the weights per parameter and level. Uses information stored in the neural_lam config files -> Important to pass the dataset_type argument! Done via create_parameter_weights.py

Example scripts to do the preprocessing are found in this folder.