from typing import Dict
from typing import Tuple

import numpy as np

from hal.data.constants import PLAYER_INPUT_FEATURES_TO_EMBED
from hal.data.constants import PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE
from hal.data.constants import PLAYER_INPUT_FEATURES_TO_NORMALIZE
from hal.data.constants import PLAYER_POSITION
from hal.data.constants import STAGE
from hal.data.constants import VALID_PLAYERS
from hal.data.normalize import NORMALIZATION_FN_BY_FEATURE
from hal.data.normalize import NormalizationFn
from hal.data.stats import FeatureStats
from hal.training.zoo.preprocess.registry import InputPreprocessRegistry
from hal.training.zoo.preprocess.registry import Player

V0_INPUT_FEATURES_BY_CATEGORY: Dict[str, Tuple[str, ...]] = {
    "player_numeric": tuple(
        PLAYER_INPUT_FEATURES_TO_NORMALIZE + PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE + PLAYER_POSITION
    ),
    "player_categorical": tuple(PLAYER_INPUT_FEATURES_TO_EMBED),
    "categorical": STAGE,
}


def _preprocess_numeric_features(
    sample: Dict[str, np.ndarray], player: str, opponent: str, stats: Dict[str, FeatureStats]
) -> np.ndarray:
    """Preprocess numeric features for both players."""
    numeric_features = V0_INPUT_FEATURES_BY_CATEGORY["player_numeric"]
    numeric_inputs = []
    for feature in numeric_features:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE[feature]
        for p in [player, opponent]:
            feature_name = f"{p}_{feature}"
            numeric_inputs.append(preprocess_fn(sample[feature_name], stats[feature_name]))  # pylint: disable=E1102
    return np.stack(numeric_inputs, axis=-1)


def _preprocess_categorical_features(
    sample: Dict[str, np.ndarray], player: Player, opponent: Player, stats: Dict[str, FeatureStats]
) -> Dict[str, np.ndarray]:
    """Preprocess categorical features for both players."""

    def process_feature(feature_name: str, column_name: str) -> np.ndarray:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE[feature_name]
        return preprocess_fn(sample[column_name], stats[column_name])[..., np.newaxis]

    processed_features = {}

    for feature in V0_INPUT_FEATURES_BY_CATEGORY["player_categorical"]:
        for p, prefix in [(player, "ego"), (opponent, "opponent")]:
            col_name = f"{p}_{feature}"  # e.g. "p1_character"
            perspective_feature_name = f"{prefix}_{feature}"  # e.g. "ego_character"
            processed_features[perspective_feature_name] = process_feature(feature, col_name)

    for feature in V0_INPUT_FEATURES_BY_CATEGORY["categorical"]:
        processed_features[feature] = process_feature(feature, column_name=feature)

    return processed_features


@InputPreprocessRegistry.register("inputs_v0")
def preprocess_inputs_v0(
    sample: Dict[str, np.ndarray], input_len: int, player: Player, stats: Dict[str, FeatureStats]
) -> Dict[str, np.ndarray]:
    """Slice input sample to the input length."""
    assert player in VALID_PLAYERS
    opponent = "p2" if player == "p1" else "p1"

    input_sample = {k: v[:input_len] for k, v in sample.items()}

    categorical_features = _preprocess_categorical_features(input_sample, player, opponent, stats)
    gamestate = _preprocess_numeric_features(input_sample, player, opponent, stats)

    return {"gamestate": gamestate, **categorical_features}
