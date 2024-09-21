# %%
from pathlib import Path

import torch
from tensordict import TensorDict

from hal.training.io import load_model_from_artifact_dir

# %%
artifact_dir = Path(
    "/opt/projects/hal2/runs/2024-09-19_08-04-20/arch@GPTv1-12-4_local_batch_size@32_n_samples@1048576"
)
model, train_config = load_model_from_artifact_dir(artifact_dir)
device = "cuda:7"
model = model.to(device)
model.train()
batch = TensorDict.load(artifact_dir / "training_samples" / "0")
batch = batch.to(device)

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
heads = ["buttons", "main_stick", "c_stick"]

x = batch["inputs"]
y = batch["targets"]

criterion = torch.nn.CrossEntropyLoss()

for i in range(1000):
    optimizer.zero_grad()
    y_hat = model(x)  # y_hat["buttons"] shape: (B, T, C)
    B, T, C = y_hat["buttons"].shape

    logits = y_hat["buttons"]
    targets = y["buttons"]

    loss = criterion(logits, targets)
    print(f"Iteration {i}, Loss: {loss.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

# %%

# %%
y_hat = model(x)
y_hat

# %%
y["buttons"][0]
# %%
y_hat["buttons"][0]

# %%
criterion(y_hat["buttons"][0], y["buttons"][0])
# %%
B, T, num_buttons = y_hat["buttons"].shape
logits = y_hat["buttons"].reshape(B * T, num_buttons)
logits.shape
# %%
y["buttons"].shape
