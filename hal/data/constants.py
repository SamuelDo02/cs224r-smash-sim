from melee import Action
from melee import Character
from melee import Stage

EXCLUDED_STAGES = ("NO_STAGE", "RANDOM_STAGE")
IDX_BY_STAGE = {stage: i for i, stage in enumerate(stage for stage in Stage if stage.name not in EXCLUDED_STAGES)}
STAGE_BY_IDX = {i: stage.name for stage, i in IDX_BY_STAGE.items()}

EXCLUDED_CHARACTERS = ("NANA", "WIREFRAME_MALE", "WIREFRAME_FEMALE", "GIGA_BOWSER", "SANDBAG", "UNKNOWN_CHARACTER")
IDX_BY_CHARACTER = {
    char: i for i, char in enumerate(char for char in Character if char.name not in EXCLUDED_CHARACTERS)
}
CHARACTER_BY_IDX = {i: char.name for char, i in IDX_BY_CHARACTER.items()}

IDX_BY_ACTION = {action: i for i, action in enumerate(Action)}
ACTION_BY_IDX = {i: action.name for action, i in IDX_BY_ACTION.items()}
