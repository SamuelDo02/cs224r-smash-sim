import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Dict

import melee
from loguru import logger
from melee import enums
from melee.menuhelper import MenuHelper

from hal.eval.emulator_paths import REMOTE_DOLPHIN_HOME_PATH
from hal.eval.emulator_paths import REMOTE_EMULATOR_PATH
from hal.eval.emulator_paths import REMOTE_EVAL_REPLAY_DIR


def get_console_kwargs(no_gui: bool = True) -> Dict[str, Any]:
    headless_console_kwargs = (
        {
            "gfx_backend": "Null",
            "disable_audio": True,
            "use_exi_inputs": True,
            "enable_ffw": True,
        }
        if no_gui
        else {}
    )
    emulator_path = REMOTE_EMULATOR_PATH
    dolphin_home_path = REMOTE_DOLPHIN_HOME_PATH
    Path(dolphin_home_path).mkdir(exist_ok=True, parents=True)
    replay_dir = REMOTE_EVAL_REPLAY_DIR
    Path(replay_dir).mkdir(exist_ok=True, parents=True)
    console_kwargs = {
        "path": emulator_path,
        "is_dolphin": True,
        "dolphin_home_path": dolphin_home_path,
        "tmp_home_directory": False,
        "replay_dir": replay_dir,
        "blocking_input": True,
        **headless_console_kwargs,
    }
    return console_kwargs


def self_play_menu_helper(
    gamestate: melee.GameState,
    controller_1: melee.Controller,
    controller_2: melee.Controller,
    character_1: melee.Character,
    character_2: melee.Character,
    stage_selected: melee.Stage,
) -> None:
    if gamestate.menu_state == enums.Menu.MAIN_MENU:
        MenuHelper.choose_versus_mode(gamestate=gamestate, controller=controller_1)
    # If we're at the character select screen, choose our character
    elif gamestate.menu_state == enums.Menu.CHARACTER_SELECT:
        player_1 = gamestate.players[controller_1.port]
        player_1_character_selected = player_1.character == character_1

        if not player_1_character_selected:
            MenuHelper.choose_character(
                character=character_1,
                gamestate=gamestate,
                controller=controller_1,
                cpu_level=0,
                costume=0,
                swag=False,
                start=False,
            )
        else:
            MenuHelper.choose_character(
                character=character_2,
                gamestate=gamestate,
                controller=controller_2,
                cpu_level=9,
                costume=1,
                swag=False,
                start=True,
            )
    # If we're at the stage select screen, choose a stage
    elif gamestate.menu_state == enums.Menu.STAGE_SELECT:
        MenuHelper.choose_stage(
            stage=stage_selected, gamestate=gamestate, controller=controller_1, character=character_1
        )
    # If we're at the postgame scores screen, spam START
    elif gamestate.menu_state == enums.Menu.POSTGAME_SCORES:
        MenuHelper.skip_postgame(controller=controller_1)


@contextmanager
def console_manager(console: melee.Console):
    def signal_handler(sig, frame):
        raise KeyboardInterrupt

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        yield
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    finally:
        signal.signal(signal.SIGINT, original_handler)
        console.stop()
        logger.info("Shutting down cleanly...")
