# %%
import torch

from hal.data.constants import ACTION_BY_IDX
from hal.data.dataset import MmappedParquetDataset

dataset_path = "/opt/projects/hal2/data/dev/train.parquet"
stats_path = "/opt/projects/hal2/data/dev/stats.json"
dataset = MmappedParquetDataset(input_path=dataset_path, stats_path=stats_path, input_len=60, target_len=10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

# %%
dataloader_iter = iter(dataloader)
inputs, targets = next(dataloader_iter)

inputs

# %%
gamestate = inputs["gamestate"]
gamestate[2, :30]

# %%
action = inputs["action_other"]
action[2, :30]
# %%
for idx in action[1, :30]:
    print(ACTION_BY_IDX[idx])
# %%
