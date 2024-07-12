from typing import Callable
from typing import Dict
from typing import Final
from typing import Tuple

import numpy as np

from hal.data.stats import FeatureStats

VALID_PLAYERS: Final[Tuple[str, str]] = ("p1", "p2")

STAGE: Tuple[str, ...] = ("stage",)
PLAYER_INPUT_FEATURES_TO_EMBED: Tuple[str, ...] = ("character", "action")
PLAYER_INPUT_FEATURES_TO_NORMALIZE: Tuple[str, ...] = (
    "percent",
    "stock",
    "facing",
    "action_frame",
    "invulnerable",
    "jumps_left",
    "on_ground",
)
PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE: Tuple[str, ...] = ("shield_strength",)
PLAYER_POSITION: Tuple[str, ...] = (
    "position_x",
    "position_y",
)
# Optional input features
PLAYER_HITLAG_FEATURES: Tuple[str, ...] = (
    "hitlag_left",
    "hitstun_left",
)
PLAYER_SPEED_FEATURES: Tuple[str, ...] = (
    "speed_air_x_self",
    "speed_y_self",
    "speed_x_attack",
    "speed_y_attack",
    "speed_ground_x_self",
)
PLAYER_ECB_FEATURES: Tuple[str, ...] = (
    "ecb_bottom_x",
    "ecb_bottom_y",
    "ecb_top_x",
    "ecb_top_y",
    "ecb_left_x",
    "ecb_left_y",
    "ecb_right_x",
    "ecb_right_y",
)
# Target features
TARGET_FEATURES_TO_ONE_HOT_ENCODE: Tuple[str, ...] = (
    "button_a",
    "button_b",
    "button_x",
    "button_z",
    "button_l",
)


NormalizationFn = Callable[[np.ndarray, FeatureStats], np.ndarray]


def cast_int32(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Cast to int32."""
    return array.astype(np.int32)


def normalize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Normalize feature [0, 1]."""
    return ((array - stats.min) / (stats.max - stats.min)).astype(np.float32)


def invert_and_normalize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Invert and normalize feature to [0, 1]."""
    return ((stats.max - array) / (stats.max - stats.min)).astype(np.float32)


def standardize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Standardize feature to mean 0 and std 1."""
    return ((array - stats.mean) / stats.std).astype(np.float32)


def union(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Perform logical OR of two features."""
    return array_1 | array_2


NORMALIZATION_FN_BY_FEATURE: Dict[str, NormalizationFn] = {
    **dict.fromkeys(STAGE, cast_int32),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_EMBED, cast_int32),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_NORMALIZE, normalize),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
    **dict.fromkeys(PLAYER_POSITION, standardize),
    **dict.fromkeys(PLAYER_HITLAG_FEATURES, normalize),
    **dict.fromkeys(PLAYER_SPEED_FEATURES, standardize),
    **dict.fromkeys(PLAYER_ECB_FEATURES, standardize),
}
