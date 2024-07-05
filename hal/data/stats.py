import json

import attr


@attr.s(auto_attribs=True, frozen=True)
class FeatureStats:
    """Contains mean, std, median, min, and max for each feature."""

    mean: float
    std: float
    min: float
    max: float
    median: float


@attr.s(auto_attribs=True, frozen=True)
class DatasetStats:
    """Contains the statistics for each feature in the dataset."""

    features: dict[str, FeatureStats]


def load_dataset_stats(path: str) -> DatasetStats:
    """Load the dataset statistics from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = {k: FeatureStats(**v) for k, v in data.items()}
    return DatasetStats(features=features)
