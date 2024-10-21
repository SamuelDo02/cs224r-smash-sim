# %%
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.config import TrainConfig
from hal.training.streaming_dataloader import get_dataloaders

config = TrainConfig(
    n_gpus=1,
    debug=True,
    arch="",
    data=DataConfig(data_dir="/opt/projects/hal2/data/mang0"),
    embedding=EmbeddingConfig(),
)
train_loader, val_loader = get_dataloaders(config)

# %%
for batch in train_loader:
    print(batch)
    break
