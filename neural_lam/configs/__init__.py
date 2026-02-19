from .ukesm import UKESMConfig
from .era5_nextgems import ERA5_NextGEMSConfig
from .era5_nextgems_minimal import ERA5_NextGEMSConfigMinimal

def get_constants(dataset_name: str):
    dataset_name = dataset_name.lower()
    for (key, val) in POSSIBLE_DATASET_TYPES.items():
        if key == dataset_name:
            return val
    raise ValueError(f"Unknown dataset: {dataset_name}")
    # if dataset_name in ["ukesm"]:
    #     return UKESMConfig()
    # elif dataset_name in ["nextgems_minimal", "era5_minimal"]:
    #     return ERA5_NextGEMSConfigMinimal()
    # elif dataset_name in ["nextgems", "era5"]:
    #     return ERA5_NextGEMSConfig()
    # else:
    #     raise ValueError(f"Unknown dataset: {dataset_name}")


POSSIBLE_DATASET_TYPES = {"ukesm": UKESMConfig(),
                          "nextgems_minimal":ERA5_NextGEMSConfigMinimal(),
                          "era5_minimal":ERA5_NextGEMSConfigMinimal(),
                          "nextgems":ERA5_NextGEMSConfig(),
                          "era5":ERA5_NextGEMSConfig()}