from typing import Dict
from typing import Final
from typing import Literal
from typing import Tuple

import numpy as np
from melee import Action
from melee import Character
from melee import Stage

NP_MASK_VALUE: Final[int] = (1 << 31) - 1

VALID_PLAYERS: Final[Tuple[str, str]] = ("p1", "p2")
Player = Literal["p1", "p2"]
PLAYER_1_PORT: Final[int] = 1
PLAYER_2_PORT: Final[int] = 2


def get_opponent(player: Player) -> Player:
    return "p2" if player == "p1" else "p1"


###################
# Gamestate      #
###################

INCLUDED_STAGES: Tuple[str, ...] = (
    "FINAL_DESTINATION",
    "BATTLEFIELD",
    "POKEMON_STADIUM",
    "DREAMLAND",
    "FOUNTAIN_OF_DREAMS",
    "YOSHIS_STORY",
)
IDX_BY_STAGE: Dict[Stage, int] = {
    stage: i for i, stage in enumerate(stage for stage in Stage if stage.name in INCLUDED_STAGES)
}
IDX_BY_STAGE_STR: Dict[str, int] = {stage.name: i for stage, i in IDX_BY_STAGE.items()}
STAGE_BY_IDX: Dict[int, str] = {i: stage.name for stage, i in IDX_BY_STAGE.items()}

INCLUDED_CHARACTERS: Tuple[str, ...] = (
    "MARIO",
    "FOX",
    "CPTFALCON",
    "DK",
    "KIRBY",
    "BOWSER",
    "LINK",
    "SHEIK",
    "NESS",
    "PEACH",
    "POPO",
    "NANA",
    "PIKACHU",
    "SAMUS",
    "YOSHI",
    "JIGGLYPUFF",
    "MEWTWO",
    "LUIGI",
    "MARTH",
    "ZELDA",
    "YLINK",
    "DOC",
    "FALCO",
    "PICHU",
    "GAMEANDWATCH",
    "GANONDORF",
    "ROY",
)
IDX_BY_CHARACTER: Dict[Character, int] = {
    char: i for i, char in enumerate(char for char in Character if char.name in INCLUDED_CHARACTERS)
}
IDX_BY_CHARACTER_STR: Dict[str, int] = {char.name: i for char, i in IDX_BY_CHARACTER.items()}
CHARACTER_BY_IDX: Dict[int, str] = {i: char.name for char, i in IDX_BY_CHARACTER.items()}

IDX_BY_ACTION: Dict[Action, int] = {action: i for i, action in enumerate(Action)}
ACTION_BY_IDX: Dict[int, str] = {i: action.name for action, i in IDX_BY_ACTION.items()}

ORIGINAL_BUTTONS: Tuple[str, ...] = (
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Y",
    "BUTTON_Z",
    "BUTTON_L",
    "BUTTON_R",
)
ORIGINAL_BUTTONS_NO_SHOULDER: Tuple[str, ...] = (
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Y",
    "BUTTON_Z",
    "NO_BUTTON",
)
INCLUDED_BUTTONS: Tuple[str, ...] = (
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Z",
    "BUTTON_L",
    "NO_BUTTON",
)
INCLUDED_BUTTONS_NO_SHOULDER: Tuple[str, ...] = (
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Z",
    "NO_BUTTON",
)


###################
# Embeddings      #
###################

REPLAY_UUID: Tuple[str] = ("replay_uuid",)
FRAME: Tuple[str] = ("frame",)
STAGE: Tuple[str, ...] = ("stage",)
PLAYER_INPUT_FEATURES_TO_EMBED: Tuple[str, ...] = ("character", "action")
PLAYER_INPUT_FEATURES_TO_NORMALIZE: Tuple[str, ...] = (
    "percent",
    "stock",
    "facing",
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
PLAYER_ACTION_FRAME_FEATURES: Tuple[str, ...] = (
    "action_frame",
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
    "a",
    "b",
    "x",
    "z",
    "l",
    "no_button",
)

SHOULDER_CLUSTER_CENTERS_V0: np.ndarray = np.array([0.0, 0.4, 1.0])
SHOULDER_CLUSTER_CENTERS_V0.flags.writeable = False

SHOULDER_CLUSTER_CENTERS_V1: np.ndarray = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
SHOULDER_CLUSTER_CENTERS_V1.flags.writeable = False

SHOULDER_CLUSTER_CENTERS_V2: np.ndarray = np.array([0.0, 0.35, 0.6, 0.85, 1.0])
SHOULDER_CLUSTER_CENTERS_V2.flags.writeable = False

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
STICK_XY_CLUSTER_CENTERS_V0.flags.writeable = False

STICK_XY_CLUSTER_CENTERS_V0_1: np.ndarray = np.array(
    [
        [0.5, 0.5],
        [1.0, 0.5],
        [0.0, 0.5],
        [0.50, 0.0],
        [0.50, 1.0],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.85, 0.85],
        [0.15, 0.85],
    ]
)
STICK_XY_CLUSTER_CENTERS_V0_1.flags.writeable = False

STICK_XY_CLUSTER_CENTERS_V1: np.ndarray = np.array(
    [
        [0.0, 0.5],
        [0.04031356, 0.67579186],
        [0.04306322, 0.32213718],
        [0.07422757, 0.74398047],
        [0.08753323, 0.23118582],
        [0.09649086, 0.5],
        [0.10760637, 0.79640961],
        [0.13590235, 0.16644649],
        [0.16092533, 0.5],
        [0.16233712, 0.85917079],
        [0.1661272, 0.13657573],
        [0.2414301, 0.08239827],
        [0.25649357, 0.28935188],
        [0.26877457, 0.71068704],
        [0.27118719, 0.5],
        [0.29513732, 0.94317985],
        [0.30770162, 0.04870167],
        [0.32365757, 0.5],
        [0.5, 0.0],
        [0.5, 0.11190532],
        [0.5, 0.21006109],
        [0.5, 0.31441873],
        [0.5, 0.5],
        [0.5, 0.67654097],
        [0.5, 0.76718795],
        [0.5, 0.86723614],
        [0.5, 1.0],
        [0.67634243, 0.5],
        [0.69229841, 0.04870167],
        [0.70486271, 0.94317985],
        [0.72881281, 0.5],
        [0.73122543, 0.71068704],
        [0.74350643, 0.28935188],
        [0.7585699, 0.08239827],
        [0.8338728, 0.13657573],
        [0.83766288, 0.85917079],
        [0.83907467, 0.5],
        [0.86409765, 0.16644649],
        [0.89239365, 0.79640961],
        [0.90350914, 0.5],
        [0.91246676, 0.23118582],
        [0.92577243, 0.74398047],
        [0.95693678, 0.32213718],
        [0.95968646, 0.67579186],
        [1.0, 0.5],
    ],
    dtype=np.float32,
)
STICK_XY_CLUSTER_CENTERS_V1.flags.writeable = False


STICK_XY_CLUSTER_CENTERS_V2 = (
    np.array(
        [  # neutral
            [0.0, 0.0],
            # partial tilt
            [0.35, 0.0],
            [-0.35, 0.0],
            [0.0, 0.35],
            [0.0, -0.35],
            # tilt
            [0.675, 0.0],
            [-0.675, 0.0],
            [0.0, 0.675],
            [0.0, -0.675],
            # full press (dash / smash attack)
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            # 17º / perfect wave/ledgedash
            [0.95, -0.3],
            [-0.95, -0.3],
            # 17º
            [0.95, 0.3],
            [-0.95, 0.3],
            # 30º / downward/up-angled f-smash
            [0.85, -0.5],
            [0.85, 0.5],
            [-0.85, -0.5],
            [-0.85, 0.5],
            # 45º + shield drops
            [0.7, -0.7],
            [-0.7, -0.7],
            [0.7, 0.7],
            [-0.7, 0.7],
            # [0.675, -0.675],
            # [-0.675, -0.675],
            # [0.675, 0.675],
            # [-0.675, 0.675],
            # up-/down-angled f-tilts
            [0.5, 0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [-0.5, -0.5],
            # 60º
            [0.5, 0.85],
            [-0.5, 0.85],
            [0.5, -0.85],
            [-0.5, -0.85],
            # 72.5º
            [0.3, -0.95],
            [0.3, 0.95],
            [-0.3, -0.95],
            [-0.3, 0.95],
        ]
    )
    / 2
    + 0.5
)
STICK_XY_CLUSTER_CENTERS_V2.flags.writeable = False


STICK_XY_CLUSTER_CENTERS_V3 = np.array(
    [
        [0.0, 0.0],
        # tilts
        [0.3, 0.0],
        [-0.3, 0.0],
        [0.0, 0.3],
        [0.0, -0.3],
        [0.4, 0.0],
        [-0.4, 0.0],
        [0.0, 0.4],
        [0.0, -0.4],
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.5],
        [0.0, -0.5],
        [0.6, 0.0],
        [-0.6, 0.0],
        [0.0, 0.6],
        [0.0, -0.6],
        [0.7, 0.0],
        [-0.7, 0.0],
        [0.0, 0.7],
        [0.0, -0.7],
        [0.8, 0.0],
        [-0.8, 0.0],
        [0.0, 0.8],
        [0.0, -0.8],
        [0.9, 0.0],
        [-0.9, 0.0],
        [0.0, 0.9],
        [0.0, -0.9],
        # full press (dash / smash attack)
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        # 16.84º / perfect wavedash
        [0.95, -0.2875],
        [-0.95, -0.2875],
        [0.95, 0.2875],
        [-0.95, 0.2875],
        # 21º
        [0.93, -0.350],
        [-0.93, -0.350],
        [0.93, 0.350],
        [-0.93, 0.350],
        # 25º
        [0.9, -0.4125],
        [-0.9, -0.4125],
        [0.9, 0.4125],
        [-0.9, 0.4125],
        # 30º / downward/up-angled f-smash
        [0.8625, -0.5],
        [0.8625, 0.5],
        [-0.8625, -0.5],
        [-0.8625, 0.5],
        # 35º
        [0.8125, -0.5750],
        [0.8125, 0.5750],
        [-0.8125, -0.5750],
        [-0.8125, 0.5750],
        # 40º
        [0.7625, -0.6375],
        [-0.7625, -0.6375],
        [0.7625, 0.6375],
        [-0.7625, 0.6375],
        # 45º + shield drops
        [0.7125, -0.7125],
        [-0.7125, -0.7125],
        [0.7125, 0.7125],
        [-0.7125, 0.7125],
        # 50º
        [0.6375, -0.7625],
        [0.6375, 0.7625],
        [-0.6375, -0.7625],
        [-0.6375, 0.7625],
        # 55º
        [0.5750, -0.8125],
        [0.5750, 0.8125],
        [-0.5750, -0.8125],
        [-0.5750, 0.8125],
        # 60º
        [0.5, 0.8625],
        [-0.5, 0.8625],
        [0.5, -0.8625],
        [-0.5, -0.8625],
        # 65º
        [0.4125, -0.9],
        [0.4125, 0.9],
        [-0.4125, -0.9],
        [-0.4125, 0.9],
        # 70º
        [0.35, -0.93],
        [0.35, 0.93],
        [-0.35, -0.93],
        [-0.35, 0.93],
        # 75º
        [0.2875, -0.95],
        [0.2875, 0.95],
        [-0.2875, -0.95],
        [-0.2875, 0.95],
        # mid quadrant
        [0.6875, -0.35],
        [0.6875, 0.35],
        [-0.6875, -0.35],
        [-0.6875, 0.35],
        [0.35, -0.6875],
        [0.35, 0.6875],
        [-0.35, -0.6875],
        [-0.35, 0.6875],
        [0.45, -0.45],
        [0.45, 0.45],
        [-0.45, -0.45],
        [-0.45, 0.45],
    ]
)
STICK_XY_CLUSTER_CENTERS_V3.flags.writeable = False
