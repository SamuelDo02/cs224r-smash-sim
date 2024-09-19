# %%
from pathlib import Path

import torch
from tensordict import TensorDict

from hal.training.io import load_model_from_artifact_dir

# %%
artifact_dir = Path("/opt/projects/hal2/runs/2024-09-19_07-51-09/arch@GPTv1-4-4_local_batch_size@32_n_samples@1048576")
model, train_config = load_model_from_artifact_dir(artifact_dir)
model.train()
batch = TensorDict.load(artifact_dir / "training_samples" / "0")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
heads = ["buttons", "main_stick", "c_stick"]

x = batch["inputs"]
y = batch["targets"]

for i in range(1000):
    optimizer.zero_grad()
    y_hat = model(x)
    # total_loss = 0
    # for head in heads:
    #     loss = criterion(y_hat[head], y[head])
    #     total_loss += loss
    # print(total_loss.item())
    # total_loss.backward()
    loss = criterion(y_hat["buttons"], y["buttons"])
    print(loss.item())
    loss.backward()
    optimizer.step()

# %%


# %%
