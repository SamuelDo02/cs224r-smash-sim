# %%
from pathlib import Path

from hal.training.io import load_model_from_artifact_dir

artifact_dir = Path("/opt/projects/hal2/runs/2024-09-04_09-18-25/arch@MLPv1_local_batch_size@256_n_samples@262144")
model = load_model_from_artifact_dir(artifact_dir)

# %%
model
