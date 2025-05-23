# %%
from pathlib import Path

import torch
from tensordict import TensorDict

from hal.training.config import DataConfig

# from hal.training.config import EmbeddingConfig
from hal.training.streaming_dataset import HALStreamingDataset

# %%
mds_dir = Path("/opt/projects/hal2/data/mang0/train")
data_config = DataConfig(
    data_dir="/opt/projects/hal2/data/mang0",
    seq_len=256,
)
# embedding_config = EmbeddingConfig(
#     input_preprocessing_fn="inputs_v0_controller",
# )
train_dataset = HALStreamingDataset(
    local=str(mds_dir),
    remote=None,
    batch_size=1,
    shuffle=False,
    data_config=data_config,
    # embedding_config=embedding_config,
    debug=True,
)
x_train = train_dataset[0]["inputs"].unsqueeze(0)
y_train = train_dataset[0]["targets"].unsqueeze(0)

# %%
x_train.save("/tmp/mang0_debugging/refactor_model_inputs_000000/")
y_train.save("/tmp/mang0_debugging/refactor_model_outputs_000000/")
# %%
x_main = TensorDict.load("/tmp/mang0_debugging/model_inputs_000000/")
y_main = TensorDict.load("/tmp/mang0_debugging/model_outputs_000000/")
# %%
for k, v in x_main.items():
    print(f"{k}: original {v.shape} vs refactored {x_train[k].shape}")

# %%
x_main["gamestate"][0, :20]
# %%
for k, v in x_main.items():
    old_tensor = v
    if k == "gamestate":
        old_tensor = v[:, :-1, :]
    # check if v matches x_train[k]
    if not torch.allclose(old_tensor, x_train[k]):
        # print the difference
        print(f"{k}: {old_tensor.shape} vs {x_train[k].shape}")
        print(f"diff: {old_tensor - x_train[k]}")
# %%
