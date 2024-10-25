# %%
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from hal.training.io import load_model_from_artifact_dir

# %%
N = 100
model_dir = (
    "/opt/projects/hal2/runs/2024-10-22_15-31-30/arch@GPTv1-12-4-dropout_local_batch_size@64_n_samples@16777216"
)
model, train_config = load_model_from_artifact_dir(Path(model_dir))
model.eval()


# %%
def mock_model() -> TensorDict:
    tensor_dict = TensorDict(
        {
            "stage": torch.randint(1, 6, (1, 256, 1)),
            "ego_character": torch.randint(0, 20, (1, 256, 1)),
            "opponent_character": torch.randint(0, 20, (1, 256, 1)),
            "ego_action": torch.randint(0, 300, (1, 256, 1)),
            "opponent_action": torch.randint(0, 300, (1, 256, 1)),
            "gamestate": torch.randn(1, 256, 18),
        },
        batch_size=(1, 256),
    )
    return tensor_dict


# %%
tensor_dicts = tuple(mock_model() for _ in range(N))
print(tensor_dicts[0])

# %%
t0 = time.perf_counter()
with torch.no_grad():
    for tensor_dict in tensor_dicts:
        model(tensor_dict)
t1 = time.perf_counter()
cpu_time = (t1 - t0) / N
print(f"Time taken (cpu): {cpu_time} seconds")

# %%
model = model.to("cuda")

# %%
t0 = time.perf_counter()
with torch.no_grad():
    for tensor_dict in tensor_dicts:
        tensor_dict = tensor_dict.to("cuda", non_blocking=False)
        model(tensor_dict)
        tensor_dict = tensor_dict.to("cpu", non_blocking=False)
t1 = time.perf_counter()
cuda_time = (t1 - t0) / N
print(f"Time taken (cuda): {cuda_time} seconds")

# %%
print("Compiling model...")
opt_model = torch.compile(model)

# %%
