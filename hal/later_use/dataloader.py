from pathlib import Path
from typing import Optional

import pyarrow
import pyarrow.dataset as ds


def get_parquet_dataset(dataset_path: Path) -> ds.Dataset:
    dataset = ds.dataset(source=dataset_path, format='parquet')
    return dataset


def filter_dataset(dataset: ds.Dataset,
                   stage: Optional[str] = None,
                   ego_char: Optional[str] = None,
                   opponent_char: Optional[str] = None,
                   ego_nickname: Optional[str] = None) -> pyarrow.Table:

    if stage is not None:
        filter_condition = (ds.field('stage') > 2)
