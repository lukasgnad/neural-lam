from .ukesm import UKESMConfig
from .era5_nextgems import ERA5_NextGEMSConfig


def get_constants(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name in ["ukesm"]:
        return UKESMConfig()
    elif dataset_name in ["nextgems", "era5"]:
        return ERA5_NextGEMSConfig()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
