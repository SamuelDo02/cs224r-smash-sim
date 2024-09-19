# %%
from pathlib import Path

from tensordict import TensorDict

from hal.training.io import load_model_from_artifact_dir

artifact_dir = Path(
    "/opt/projects/hal2/runs/2024-09-19_06-18-33/arch@MLPDebug-512-8_local_batch_size@32_n_samples@1048576"
)

model, train_config = load_model_from_artifact_dir(artifact_dir)
model.eval()

# %%
batch = TensorDict.load(artifact_dir / "training_samples" / "0")

# %%
batch["inputs"].shape

# %%
batch["targets"].shape

# %%
x = batch["inputs"][0].unsqueeze(0)
y = batch["targets"][0].unsqueeze(0)

# %%
