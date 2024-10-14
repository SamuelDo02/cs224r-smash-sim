from typing import Tuple

from streaming import StreamingDataset
from torch.utils.data import DataLoader

from hal.training.config import TrainConfig


def get_dataloaders(config: TrainConfig, rank: int, world_size: int) -> Tuple[DataLoader, DataLoader]:
    train = StreamingDataset(config.data.data_dir)
