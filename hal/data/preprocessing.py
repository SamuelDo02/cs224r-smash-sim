# %%
from typing import Dict

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from hal.data.stats import FeatureStats
from hal.data.stats import load_dataset_stats

np.set_printoptions(threshold=np.inf)


INPUT_FEATURES_TO_EMBED = ("stage", "character", "action")
INPUT_FEATURES_TO_NORMALIZE = ("percent", "stock", "facing", "action_frame", "invulnerable", "jumps_left", "on_ground")
INPUT_FEATURES_TO_INVERT_AND_NORMALIZE = ("shield_strength",)
INPUT_FEATURES_TO_STANDARDIZE = (
    "position_x",
    "position_y",
    "hitlag_left",
    "hitstun_left",
    "speed_air_x_self",
    "speed_y_self",
    "speed_x_attack",
    "speed_y_attack",
    "speed_ground_x_self",
)

TARGET_FEATURES_TO_ONE_HOT_ENCODE = ("button_a", "button_b", "button_x", "button_z", "button_l")


def pyarrow_table_to_np_dict(table: pa.Table) -> Dict[str, np.ndarray]:
    """Convert pyarrow table to dictionary of numpy arrays."""
    return {name: col.to_numpy() for name, col in zip(table.column_names, table.columns)}


def normalize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Normalize feature [0, 1]."""
    return (array - stats.min) / (stats.max - stats.min)


def invert_and_normalize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Invert and normalize feature to [0, 1]."""
    return (stats.max - array) / (stats.max - stats.min)


def standardize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Standardize feature to mean 0 and std 1."""
    return (array - stats.mean) / stats.std


def union(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Perform logical OR of two features."""
    return array_1 | array_2


def encode_stacked_array_to_one_hot(array: np.ndarray) -> np.ndarray:
    """Convert stacked features to one hot encoding."""
    non_zero_mask = array > 0
    row_sum = np.sum(non_zero_mask, axis=1)

    one_hot = np.zeros_like(array)

    # Handle rows with single non-zero element
    single_non_zero = row_sum == 1
    one_hot[single_non_zero] = non_zero_mask[single_non_zero]

    # Handle rows with multiple non-zero elements
    multi_non_zero = row_sum > 1
    if np.any(multi_non_zero):
        # Find the index of the last non-zero element in each row
        last_non_zero_indices = np.where(non_zero_mask[multi_non_zero])[1].reshape(
            -1, row_sum[multi_non_zero][:, None]
        )[:, -1]
        one_hot[multi_non_zero, last_non_zero_indices] = 1

    return one_hot


feature_processors = {
    INPUT_FEATURES_TO_EMBED: lambda x: x,
    INPUT_FEATURES_TO_NORMALIZE: normalize,
    INPUT_FEATURES_TO_INVERT_AND_NORMALIZE: invert_and_normalize,
    INPUT_FEATURES_TO_STANDARDIZE: standardize,
}


def preprocess_features_v0(sample: Dict[str, np.ndarray], stats: Dict[str, FeatureStats]) -> Dict[str, np.ndarray]:
    """Preprocess features."""
    preprocessed = {}

    # Stack buttons and encode one_hot
    for player in ("p1", "p2"):
        button_a = (sample[f"{player}_button_a"]).astype(np.bool_)
        button_b = (sample[f"{player}_button_b"]).astype(np.bool_)
        button_z = sample[f"{player}_button_z"]
        jump = union(sample[f"{player}_button_x"], sample[f"{player}_button_y"])
        shoulder = union(sample[f"{player}_button_l"], sample[f"{player}_button_r"])
        no_button = np.ones_like(sample[f"{player}_button_a"])

        stacked_buttons = np.stack((button_a, button_b, button_z, jump, shoulder, no_button), axis=1)
        preprocessed[f"{player}_buttons"] = encode_stacked_array_to_one_hot(stacked_buttons)

    # for feature_list, preprocessing_func in feature_processors.items():
    #     for feature in feature_list:
    #         process_feature(feature, preprocessing_func)

    return preprocessed


# %%
# load dataset, load stats and apply them to the dataset
input_path = "/opt/projects/hal2/data/dev/val.parquet"
stats_path = "/opt/projects/hal2/data/dev/stats.json"

table: pa.Table = pq.read_table(input_path, memory_map=True)
stats = load_dataset_stats(stats_path)

# %%
shield = table["p1_shield_strength"].to_numpy()
shield = invert_and_normalize(shield, stats["p1_shield_strength"])
shield[:10000]

# %%
table_slice = table
preprocessed = preprocess_features_v0(pyarrow_table_to_np_dict(table_slice), stats)
# %%
a = preprocessed["p1_buttons"]
b = table_slice["p1_button_x"].to_numpy() | table_slice["p1_button_y"].to_numpy()
a[:, 3] == b
# %%
