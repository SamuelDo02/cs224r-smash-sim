# %%
import numpy as np
from constants import ACTION_BY_IDX
from tensordict import TensorDict

from hal.preprocess.transformations import convert_multi_hot_to_one_hot_early_release
from hal.preprocess.transformations import encode_original_buttons_multi_hot

# %%
buttons_LD = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)

convert_multi_hot_to_one_hot_early_release(buttons_LD)

# %%
buttons_LD = np.array(
    [
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ],
)

convert_multi_hot_to_one_hot_early_release(buttons_LD)

# %%
from streaming import StreamingDataset

mds_path = "/opt/projects/hal2/data/ranked/diamond/train"
ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)

x = ds[0]
# %%
orig = encode_original_buttons_multi_hot(TensorDict(x), "p1")
one_hot = convert_multi_hot_to_one_hot_early_release(orig.numpy())
print(f"{orig=}")
print(f"{one_hot=}")

# %%
np.where(x["p1_button_l"] == 1)
# %%
np.where(x["p1_button_r"] == 1)
# %%
multi_hot = encode_original_buttons_multi_hot(TensorDict(x), "p1")
# %%
multi_hot[862:875]
# %%
action_idx = x["p1_action"][862:875]
actions = [ACTION_BY_IDX[i] for i in action_idx]
actions
# %%
main_x, main_y = x["p1_main_stick_x"][862:875], x["p1_main_stick_y"][862:875]

# %%
x["p1_l_shoulder"][862:875].tolist()
# %%
x["p1_r_shoulder"][862:875].tolist()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Create a figure
plt.figure(figsize=(8, 8))

# Plot the main stick positions
plt.scatter(main_x, main_y, color="blue", label="Main Stick Positions")

# Draw the unit circle with radius 0.5 centered at (0.5, 0.5)
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = 0.5 + 0.5 * np.cos(theta)
circle_y = 0.5 + 0.5 * np.sin(theta)
plt.plot(circle_x, circle_y, "r--", label="Unit Circle (r=0.5)")

# Add a point at the center (0.5, 0.5)
plt.scatter(0.5, 0.5, color="red", marker="x", s=100, label="Center (0.5, 0.5)")

# Set limits, labels, and grid
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Main Stick Positions with Unit Circle (r=0.5)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()

import numpy as np

# %%
import torch
from tensordict import TensorDict

from hal.constants import SHOULDER_CLUSTER_CENTERS_V1
from hal.preprocess.transformations import get_closest_1D_clusters

x = np.array([[0.0, 0.2], [0.34, 0.7], [0.55, 0.9]])
get_closest_1D_clusters(x, SHOULDER_CLUSTER_CENTERS_V1)
# %%

from hal.preprocess.transformations import encode_shoulder_one_hot_coarse
from hal.preprocess.transformations import encode_shoulder_original_coarse

encode_shoulder_original_coarse(
    TensorDict({"p1_l_shoulder": torch.tensor([0.0, 0.34, 0.55]), "p1_r_shoulder": torch.tensor([0.4, 0.7, 0.98])}),
    "p1",
)
encode_shoulder_one_hot_coarse(
    TensorDict({"p1_l_shoulder": torch.tensor([0.0, 0.34, 0.55]), "p1_r_shoulder": torch.tensor([0.4, 0.7, 0.98])}),
    "p1",
).shape
# %%
