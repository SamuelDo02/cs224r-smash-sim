# %%
import pyarrow.parquet as pq

# table = pq.read_table("/opt/projects/hal2/data/mang0/train/17d96ec4-60e0-42b2-9a33-de3068059d64.parquet")
table = pq.read_table("/opt/projects/hal2/data/mang0/train.parquet")

# %%
unique_uuids = table["replay_uuid"].unique()
len(unique_uuids)

# %%
len(table)
