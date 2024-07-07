# %%
import torch

from hal.data.dataset import MmappedParquetDataset

dataset_path = "/opt/projects/hal2/data/dev/train.parquet"
stats_path = "/opt/projects/hal2/data/dev/stats.json"
dataset = MmappedParquetDataset(input_path=dataset_path, stats_path=stats_path, input_len=60, target_len=10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# %%
dataloader_iter = iter(dataloader)
inputs, targets = next(dataloader_iter)

num = inputs["numeric"]
cat = inputs["categorical"]

# %%
num
# %%
cat
# %%
targets
