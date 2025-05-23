# %%
from collections import Counter

import numpy as np

from hal.data.masked_dataset import MaskedDataset

# %%
ds = MaskedDataset(local="/opt/projects/hal2/data/ranked/diamond/train", batch_size=1, shuffle=True)

# %%
x = ds[0]
x = sorted(x.items(), key=lambda x: x[0])

# %%
for k, v in x:
    if v.mask.any():
        print(f"has mask: {k} min={np.min(v)} max={np.max(v)}")
    else:
        print(f"no mask: {k} min={np.min(v)} max={np.max(v)}")

# %%
for k, v in x.items():
    print(k, np.min(v), np.max(v))

# %%
from hal.constants import NP_MASK_VALUE

NP_MASK_VALUE

# %%
# find proportion for each character id that appear in the replays
character_ids = Counter()
stage_ids = Counter()

for i, x in enumerate(ds):
    if i > 10000:
        break
    if i % 1000 == 0:
        print(f"processed {i} replays")
    character_ids[x["p1_character"][0].item()] += 1
    character_ids[x["p2_character"][0].item()] += 1
    stage_ids[x["stage"][0].item()] += 1

# %%
from hal.constants import CHARACTER_BY_IDX

# graph bar chart with character ids mapped to character names
total = sum(character_ids.values())
for k, v in sorted(character_ids.items(), key=lambda x: x[1], reverse=True):
    print(f'"{CHARACTER_BY_IDX[k]}": {v / total:.3}')

# %%
from hal.constants import STAGE_BY_IDX

total = sum(stage_ids.values())
for k, v in sorted(stage_ids.items(), key=lambda x: x[1], reverse=True):
    print(f'"{STAGE_BY_IDX[k]}": {v / total:.3}')

# %%
