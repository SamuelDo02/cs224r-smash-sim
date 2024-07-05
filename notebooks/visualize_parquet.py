# %%
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import seaborn as sns
from pyarrow import parquet as pq

from hal.data.constants import ACTION_BY_IDX

np.set_printoptions(threshold=np.inf)

# %%
table: pa.Table = pq.read_table("/opt/projects/hal2/data/dev/val.parquet")

# %%
table.column_names

# %%
# randomly sample rows
table = table.take(np.random.choice(len(table), 20000, replace=False))

# %%
table["p1_position_y"].to_numpy().max()

# %%
# uuid_filter = pc.field("replay_uuid") == 5393121284994579877
# replay = table.filter(uuid_filter)

# p1_l_shoulder = replay["p1_l_shoulder"].to_pylist()
# p1_button_l = replay["p1_button_l"].to_pylist()
# for i, (analog, button) in enumerate(zip(p1_l_shoulder, p1_button_l)):
#     if math.ceil(analog) != button or math.floor(analog) != button:
#         print(f"{i=}, {analog=}, {button=}")

# %%
# p1_l_shoulder = replay["p1_l_shoulder"].to_numpy()
# p1_button_l = replay["p1_button_l"].to_numpy()
# print(f"{p1_l_shoulder.mean()=}")
# print(f"{p1_button_l.mean()=}")

# print(p1_l_shoulder)
# print(p1_button_l)

# %%
len(table)


# %%
def visualize_position_heatmap(pyarrow_table: pa.Table, x_field: str, y_field: str, title: str) -> None:
    # Extract x and y values
    x = pyarrow_table[x_field].to_numpy()
    y = pyarrow_table[y_field].to_numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a smooth heatmap using KDE
    sns.kdeplot(x=x, y=y, cmap="YlOrRd", fill=True, cbar=True, ax=ax)

    # Set labels and title
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(title)

    # Set axis limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    # Invert y-axis to match stick orientation
    ax.invert_yaxis()

    # Show the plot
    plt.show()


# %%
# plot histogram of actions
actions = table["p1_action"].to_numpy()
actions = [ACTION_BY_IDX[action] for action in actions]

# %%
# rotate text and sort
plt.hist(actions, bins=185, rotation=90, sort=True)
plt.title("Player 1 Action Histogram")
plt.show()


# %%
visualize_position_heatmap(table, "p1_position_x", "p1_position_y", "Player 1 Position Heatmap")

# # %%
visualize_position_heatmap(table, "p1_main_stick_x", "p1_main_stick_y", "Player 1 Main Stick Heatmap")
