var_rename_dict = {
    "lat": "latitude",
    "lon": "longitude",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "z": "geopotential",
    "q": "specific_humidity",
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "tp": "total_precipitation_6hr",
}

bilinear_vars = [
    "z",  # "geopotential",
    "t",  # "temperature",
    "w",  # "vertical_velocity",
    "2t",  # "2m_temperature"
    "msl",  # "mean_sea_level_pressure",
    "geopotential_at_surface",
    "u",  # "u_component_of_wind",
    "v",  # "v_component_of_wind",
    "10u",  # "10m_u_component_of_wind",
    "10v",  # "10m_v_component_of_wind",
]
conservative_vars = [
    "q",  # "specific_humidity",
    "tp",  # "total_precipitation_6hr"
]
nearest_vars = ["land_sea_mask"]
