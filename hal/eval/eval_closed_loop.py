import argparse
import signal
import sys
from collections import defaultdict
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import Optional
from typing import Sequence

import melee
import torch
from loguru import logger
from melee import enums
from melee.menuhelper import MenuHelper
from tensordict import TensorDict

from hal.data.schema import PYARROW_DTYPE_BY_COLUMN
from hal.data.stats import load_dataset_stats
from hal.eval.emulator_paths import REMOTE_CISO_PATH
from hal.eval.emulator_paths import REMOTE_DOLPHIN_HOME_PATH
from hal.eval.emulator_paths import REMOTE_EMULATOR_PATH
from hal.eval.emulator_paths import REMOTE_EVAL_REPLAY_DIR
from hal.eval.eval_helper import extract_and_append_gamestate
from hal.eval.eval_helper import send_controller_inputs
from hal.training.io import load_model_from_artifact_dir
from hal.training.preprocess.registry import InputPreprocessRegistry
from hal.training.preprocess.registry import OutputProcessingRegistry

PLAYER_1_PORT = 1
PLAYER_2_PORT = 2


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


def get_mock_framedata(seq_len: int) -> TensorDict:
    """Mock frame data for warming up compiled model."""
    return TensorDict({k: torch.zeros(seq_len) for k in PYARROW_DTYPE_BY_COLUMN}, batch_size=(seq_len,))


def convert_frame_data_to_tensor_dict(frame_data: DefaultDict[str, Sequence]) -> TensorDict:
    return TensorDict({k: torch.tensor(v) for k, v in frame_data.items()}, batch_size=(len(frame_data["frame"])))


def pad_tensors(td: TensorDict, length: int) -> TensorDict:
    """For models with fixed input length, pad with zeros.

    Assumes tensors are of shape (T, D)."""
    if td.shape[0] < length:
        pad_size = length - td.shape[0]
        return TensorDict({k: torch.nn.functional.pad(v, (pad_size, 0)) for k, v in td.items()}, batch_size=(length,))
    return td


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


def run_episode(model_dir: str, no_gui: bool = True, idx: Optional[int] = None) -> None:
    console_kwargs = get_console_kwargs(no_gui=no_gui)
    console = melee.Console(**console_kwargs)

    controller_1 = melee.Controller(console=console, port=PLAYER_1_PORT, type=melee.ControllerType.STANDARD)
    controller_2 = melee.Controller(console=console, port=PLAYER_2_PORT, type=melee.ControllerType.STANDARD)

    # Run the console
    console.run(iso_path=REMOTE_CISO_PATH, dolphin_user_path=REMOTE_DOLPHIN_HOME_PATH)
    # Connect to the console
    logger.info("Connecting to console...")
    if not console.connect():
        logger.info("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    logger.info("Console connected")

    # Plug our controller in
    #   Due to how named pipes work, this has to come AFTER running dolphin
    #   NOTE: If you're loading a movie file, don't connect the controller,
    #   dolphin will hang waiting for input and never receive it
    logger.info("Connecting controller 1 to console...")
    if not controller_1.connect():
        logger.info("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    logger.info("Controller 1 connected")
    logger.info("Connecting controller 2 to console...")
    if not controller_2.connect():
        logger.info("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    logger.info("Controller 2 connected")

    model, train_config = load_model_from_artifact_dir(Path(model_dir), idx=idx)
    model.eval()

    preprocess_inputs = InputPreprocessRegistry.get(train_config.embedding.input_preprocessing_fn)
    stats_by_feature_name = load_dataset_stats(train_config.data.stats_path)
    postprocess_outputs = OutputProcessingRegistry.get(train_config.embedding.target_preprocessing_fn)

    # TODO(eric): move to separate process w/ api
    logger.info("Compiling model...")
    model = model.to("cuda")
    model = torch.compile(model, mode="default")
    mock_tensordict = get_mock_framedata(train_config.data.input_len)
    mock_inputs = (
        preprocess_inputs(mock_tensordict, train_config.data, "p1", stats_by_feature_name).unsqueeze(0).to("cuda")
    )
    with torch.no_grad():
        model(mock_inputs)[:, -1]

    # Container for sliding window of model inputs
    frame_data: DefaultDict[str, deque] = defaultdict(lambda: deque(maxlen=train_config.data.input_len))

    # Main loop
    i = 0
    match_started = False
    with console_manager(console):
        logger.info("Starting episode")

        while i < 10000:
            gamestate = console.step()
            if gamestate is None:
                logger.info("Gamestate is None")
                break

            # The console object keeps track of how long your bot is taking to process frames
            #   And can warn you if it's taking too long
            if console.processingtime * 1000 > 12:
                logger.info("WARNING: Last frame took " + str(console.processingtime * 1000) + "ms to process.")

            # What menu are we in?
            if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                if match_started:
                    break

                self_play_menu_helper(
                    gamestate=gamestate,
                    controller_1=controller_1,
                    controller_2=controller_2,
                    character_1=melee.Character.FOX,
                    character_2=melee.Character.FOX,
                    stage_selected=melee.Stage.BATTLEFIELD,
                )
            else:
                if not match_started:
                    match_started = True

                extract_and_append_gamestate(gamestate=gamestate, frame_data=frame_data)
                frame_data_td = convert_frame_data_to_tensor_dict(frame_data)
                model_inputs = pad_tensors(frame_data_td, train_config.data.input_len)
                model_inputs = preprocess_inputs(model_inputs, train_config.data, "p1", stats_by_feature_name)
                # Unsqueeze batch dim
                model_inputs = model_inputs.unsqueeze(0).to("cuda")
                with torch.no_grad():
                    outputs: TensorDict = model(model_inputs)[:, -1].to("cpu")
                controller_inputs = postprocess_outputs(outputs)
                send_controller_inputs(controller_1, controller_inputs)

                i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Melee in emulator")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--debug", action="store_true", help="Run with debug mode")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    args = parser.parse_args()
    run_episode(model_dir=args.model_dir, no_gui=args.no_gui)
