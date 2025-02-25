# %%
import numpy as np
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

# %%
x = ds[0]
orig = encode_original_buttons_multi_hot(TensorDict(x), "p1")
one_hot = convert_multi_hot_to_one_hot_early_release(orig.numpy())
print(f"{orig=}")
print(f"{one_hot=}")

# %%
