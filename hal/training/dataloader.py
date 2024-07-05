from pathlib import Path

import pyarrow.dataset as ds
import torch


def get_parquet_dataset(dataset_path: Path) -> ds.Dataset:
    dataset = ds.dataset(source=dataset_path, format="parquet")
    return dataset


class DistributedParquetDataset(torch.data.Dataset):
    def __init__(self, dataset_path: Path) -> None:
        self.dataset = get_parquet_dataset(dataset_path)

    def __len__(self) -> int:
        return len(self.dataset)
