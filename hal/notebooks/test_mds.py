# %%
# check that mds is valid
from streaming import StreamingDataset

dataset = StreamingDataset(
    local="/opt/projects/hal2/data/ranked_dev/train",
    batch_size=1,
    split=None,
    shuffle=True,
)

# %%
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

# %%
batch = next(iter(dataloader))

# %%
batch.keys()

# %%
batch["p2_percent"].shape

# %%
batch["replay_uuid"]
