from typing import Dict
from typing import Tuple

import numpy as np
from melee import Action
from melee import Character
from melee import Stage

###################
# Gamestate      #
###################

EXCLUDED_STAGES: Tuple[str, ...] = ("NO_STAGE", "RANDOM_STAGE")
IDX_BY_STAGE: Dict[Stage, int] = {
    stage: i for i, stage in enumerate(stage for stage in Stage if stage.name not in EXCLUDED_STAGES)
}
IDX_BY_STAGE_STR: Dict[str, int] = {stage.name: i for stage, i in IDX_BY_STAGE.items()}
STAGE_BY_IDX: Dict[int, str] = {i: stage.name for stage, i in IDX_BY_STAGE.items()}

EXCLUDED_CHARACTERS: Tuple[str, ...] = (
    "NANA",
    "WIREFRAME_MALE",
    "WIREFRAME_FEMALE",
    "GIGA_BOWSER",
    "SANDBAG",
    "UNKNOWN_CHARACTER",
)
IDX_BY_CHARACTER: Dict[Character, int] = {
    char: i for i, char in enumerate(char for char in Character if char.name not in EXCLUDED_CHARACTERS)
}
IDX_BY_CHARACTER_STR: Dict[str, int] = {char.name: i for char, i in IDX_BY_CHARACTER.items()}
CHARACTER_BY_IDX: Dict[int, str] = {i: char.name for char, i in IDX_BY_CHARACTER.items()}

IDX_BY_ACTION: Dict[Action, int] = {action: i for i, action in enumerate(Action)}
ACTION_BY_IDX: Dict[int, str] = {i: action.name for action, i in IDX_BY_ACTION.items()}

INCLUDED_BUTTONS: Tuple[str, ...] = (
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Z",
    "BUTTON_L",
    "NO_BUTTON",
)


###################
# Embeddings      #
###################

STICK_XY_CLUSTER_CENTERS_V0: np.ndarray = np.array(
    [
        [0.5, 0.5],
        [1.0, 0.5],
        [0.0, 0.5],
        [0.50, 0.0],
        [0.50, 1.0],
        [0.50, 0.25],
        [0.50, 0.75],
        [0.75, 0.5],
        [0.25, 0.5],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.85, 0.85],
        [0.15, 0.85],
        [0.28, 0.93],
        [0.28, 0.07],
        [0.72, 0.07],
        [0.72, 0.93],
        [0.07, 0.28],
        [0.07, 0.72],
        [0.93, 0.72],
        [0.93, 0.28],
    ]
)
