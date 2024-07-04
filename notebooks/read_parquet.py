# %%
import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import parquet as pq

np.set_printoptions(threshold=np.inf)

# %%
table: pa.Table = pq.read_table("/opt/projects/hal2/data/train.parquet")

# %%
table.column_names

# %%
uuid_filter = pc.field("replay_uuid") == 5393121284994579877
replay = table.filter(uuid_filter)

p1_l_shoulder = replay["p1_l_shoulder"].to_pylist()
p1_button_l = replay["p1_button_l"].to_pylist()
for i, (analog, button) in enumerate(zip(p1_l_shoulder, p1_button_l)):
    if math.ceil(analog) != button or math.floor(analog) != button:
        print(f"{i=}, {analog=}, {button=}")

# %%
# p1_l_shoulder = replay["p1_l_shoulder"].to_numpy()
# p1_button_l = replay["p1_button_l"].to_numpy()
# print(f"{p1_l_shoulder.mean()=}")
# print(f"{p1_button_l.mean()=}")

print(p1_l_shoulder)
print(p1_button_l)
