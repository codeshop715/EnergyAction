from .rlbench import (
    Peract2Dataset,
    Peract2SingleCamDataset,
    Peract2BimanualDataset,
    PeractDataset,
    PeractTwoCamDataset,
    HiveformerDataset
)


DATASET_MAPPING = {
    "Peract2_3dfront_3dwrist": Peract2Dataset,
    "Peract2_3dfront": Peract2SingleCamDataset,
    "Peract": PeractDataset,
    "PeractTwoCam": PeractTwoCamDataset,
    "HiveformerRLBench": HiveformerDataset,
    "Peract2SingleCam": Peract2SingleCamDataset,
    "Peract2Bimanual": Peract2BimanualDataset,
}


def fetch_dataset_class(name):
    """Fetch the dataset class based on the dataset name."""
    if name not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {name}")
    
    return DATASET_MAPPING[name]
