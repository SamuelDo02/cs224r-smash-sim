# %%
import torch

ce = torch.nn.CrossEntropyLoss()

x = torch.tensor([0.1, -1, 0, 0, 1])
x = x.repeat(4, 2, 1)  # Shape: (B=3, T=2, C=5)
x.shape
# %%
y_one_hot = torch.tensor([0, 0, 0, 0, 1.0])  # One-hot for class 4
y_one_hot = y_one_hot.repeat(4, 2, 1)
y_one_hot.shape

# %%
ce(x, y_one_hot)

# %%
ce(x.reshape(-1, 5), y_one_hot.reshape(-1, 5))

# %%
ce(torch.tensor([0.1, -1, 0, 0, 1]), torch.tensor(4))

# %%
y_class_idx = torch.tensor(4)  # Correct class index
y_class_idx = y_class_idx.repeat(4, 2)
y_class_idx.shape

# %%
ce(x, y_class_idx)

# %%
y_class_idx = torch.tensor(4)  # Correct class index

# Let's change y_one_hot to incorrectly represent class 0
y_one_hot_incorrect = torch.tensor([1.0, 0, 0, 0, 0])  # One-hot for class 0

# Compute losses
loss_one_hot = ce(x, y_one_hot)
loss_class_idx = ce(x, y_class_idx)
loss_one_hot_incorrect = ce(x, y_one_hot_incorrect)

print(f"Loss with one-hot target: {loss_one_hot.item()}")
print(f"Loss with class index target: {loss_class_idx.item()}")
print(f"Loss with incorrect one-hot target: {loss_one_hot_incorrect.item()}")
