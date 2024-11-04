import argparse
import sys
import time
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence

import attr
import melee
import torch
import torch.multiprocessing as mp
from loguru import logger
from tensordict import TensorDict

from hal.data.schema import PYARROW_DTYPE_BY_COLUMN
from hal.data.stats import FeatureStats
from hal.data.stats import load_dataset_stats
from hal.eval.emulator_helper import console_manager
from hal.eval.emulator_helper import get_console_kwargs
from hal.eval.emulator_helper import self_play_menu_helper
from hal.eval.emulator_paths import REMOTE_CISO_PATH
from hal.eval.emulator_paths import REMOTE_DOLPHIN_HOME_PATH
from hal.eval.eval_helper import send_controller_inputs
from hal.gamestate_utils import extract_gamestate_as_tensordict
from hal.training.config import TrainConfig
from hal.training.io import load_config_from_artifact_dir
from hal.training.io import load_model_from_artifact_dir
from hal.training.preprocess.registry import InputPreprocessFn
from hal.training.preprocess.registry import InputPreprocessRegistry
from hal.training.preprocess.registry import PredPostprocessFn
from hal.training.preprocess.registry import PredPostprocessingRegistry

mp.set_start_method("spawn", force=True)

PLAYER_1_PORT = 1
PLAYER_2_PORT = 2


def get_mock_framedata(seq_len: int) -> TensorDict:
    """Mock frame data for warming up compiled model."""
    return TensorDict({k: torch.zeros(seq_len) for k in PYARROW_DTYPE_BY_COLUMN}, batch_size=(seq_len,))


def convert_frame_data_to_tensor_dict(frame_data: Mapping[str, Sequence]) -> TensorDict:
    return TensorDict({k: torch.tensor(v) for k, v in frame_data.items()}, batch_size=(len(frame_data["frame"])))


def pad_tensors(td: TensorDict, length: int) -> TensorDict:
    """For models with fixed input length, pad with zeros.

    Assumes tensors are of shape (T, D)."""
    if td.shape[0] < length:
        pad_size = length - td.shape[0]
        return TensorDict({k: torch.nn.functional.pad(v, (pad_size, 0)) for k, v in td.items()}, batch_size=(length,))
    return td


def run_episode(max_steps: int = 8 * 60 * 60) -> Generator[Optional[melee.GameState], TensorDict, None]:
    console_kwargs = get_console_kwargs()
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

    i = 0
    match_started = False
    with console_manager(console):
        logger.info("Starting episode")
        try:
            while i < max_steps:
                gamestate = console.step()
                if gamestate is None:
                    logger.info("Gamestate is None")
                    break

                if console.processingtime * 1000 > 12:
                    logger.info("WARNING: Last frame took " + str(console.processingtime * 1000) + "ms to process.")

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

                    # Yield gamestate and receive controller inputs
                    controller_inputs = yield gamestate
                    send_controller_inputs(controller_1, controller_inputs)

                    i += 1
        finally:
            # Signal end of episode
            yield None


def cpu_worker(
    shared_batched_model_input: TensorDict,
    shared_batched_model_output: TensorDict,
    rank: int,
    preprocess_inputs: InputPreprocessFn,
    postprocess_outputs: PredPostprocessFn,
    model_input_ready_flags: EventType,
    model_output_ready_flags: EventType,
    stop_event: EventType,
    train_config: TrainConfig,
    stats_by_feature_name: Dict[str, FeatureStats],
) -> None:
    """
    CPU worker that preprocesses data, writes it into shared memory,
    and reads the result after GPU processing.
    """
    emulator_generator = run_episode()
    for gamestate in emulator_generator:
        if gamestate is None:
            break

        gamestate_td = extract_gamestate_as_tensordict(gamestate)
        # Preprocess single frame
        data_config = attr.evolve(train_config.data, input_len=1, target_len=0)
        model_inputs = preprocess_inputs(gamestate_td, data_config, "p1", stats_by_feature_name)
        shared_batched_model_input[rank].copy_(model_inputs)
        model_input_ready_flags[rank].set()

        # Wait for the output to be ready
        while not model_output_ready_flags[rank].is_set() and not stop_event.is_set():
            time.sleep(0.0001)  # Sleep briefly to avoid busy waiting

        if stop_event.is_set():
            break

        # Read the output from shared_batched_model_output
        output = shared_batched_model_output[rank].clone()
        controller_inputs = postprocess_outputs(output)
        emulator_generator.send(controller_inputs)

        # Clear the output ready flag for the next iteration
        model_output_ready_flags[rank].clear()

    stop_event.set()


def gpu_worker(
    shared_batched_model_input: TensorDict,  # (n_workers,)
    shared_batched_model_output: TensorDict,  # (n_workers,)
    model_input_ready_flags: List[EventType],
    model_output_ready_flags: List[EventType],
    context_window_size: int,
    stop_event: List[EventType],
    model_dir: str,
    device: torch.device | str,
    idx: Optional[int] = None,
) -> None:
    """
    GPU worker that batches data from shared memory, updates the context window,
    performs inference with model, and writes output back to shared memory.
    """
    model, _ = load_model_from_artifact_dir(Path(model_dir), idx=idx)
    model.eval()
    model.to(device)

    # Stack along time dimension
    # shape: (n_workers, context_window_size)
    context_window: TensorDict = torch.stack([shared_batched_model_input[i] for i in range(context_window_size)], dim=-1).to(device)  # type: ignore

    # Warmup CUDA graphs with dummy inputs
    logger.info("Compiling model...")
    model = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        model(context_window)

    while not all(event.is_set() for event in stop_event):
        # Wait for all CPU workers to signal that data is ready
        for flag in model_input_ready_flags:
            while not flag.is_set() and not all(event.is_set() for event in stop_event):
                time.sleep(0.0001)  # Sleep briefly to avoid busy waiting

        if all(event.is_set() for event in stop_event):
            break

        # Read data from shared tensor on cpu
        batch_data = shared_batched_model_input.clone().to(device)  # (n_workers,)
        # Update the context window by rolling and adding new data
        context_window[:, :-1] = context_window[:, 1:].clone()
        context_window[:, -1] = batch_data
        with torch.no_grad():
            outputs: TensorDict = model(context_window)[:, -1]  # (n_workers,)
        # Write the output to shared_batched_model_output
        shared_batched_model_output.copy_(outputs)

        # Signal to CPU workers that output is ready
        for flag in model_output_ready_flags:
            flag.set()

        # Clear model_input_ready_flags for the next iteration
        for flag in model_input_ready_flags:
            flag.clear()


def main(model_dir: str, n_workers: int, idx: Optional[int] = None) -> None:
    # Set the multiprocessing start method
    mp.set_start_method("spawn")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_config: TrainConfig = load_config_from_artifact_dir(Path(model_dir))
    preprocess_inputs = InputPreprocessRegistry.get(train_config.embedding.input_preprocessing_fn)
    stats_by_feature_name = load_dataset_stats(train_config.data.stats_path)
    postprocess_outputs = PredPostprocessingRegistry.get(train_config.embedding.target_preprocessing_fn)

    # Create events to signal when cpu and gpu workers are ready
    model_input_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    model_output_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    # Create events to signal when emulator episodes end
    stop_events: List[EventType] = [mp.Event() for _ in range(n_workers)]

    # TODO: initialize
    shared_batched_model_input: Any
    shared_batched_model_output: Any

    gpu_process: mp.Process = mp.Process(
        target=gpu_worker,
        args=(
            shared_batched_model_input,
            shared_batched_model_output,
            model_input_ready_flags,
            model_output_ready_flags,
            train_config.data.input_len,
            stop_events,
            device,
            idx,
        ),
    )
    gpu_process.start()

    cpu_processes: List[mp.Process] = []
    for i in range(n_workers):
        p: mp.Process = mp.Process(
            target=cpu_worker,
            args=(
                shared_batched_model_input,
                shared_batched_model_output,
                i,
                preprocess_inputs,
                postprocess_outputs,
                model_input_ready_flags[i],
                model_output_ready_flags[i],
                stop_events[i],
                train_config,
                stats_by_feature_name,
            ),
        )
        cpu_processes.append(p)
        p.start()

    gpu_process.join()

    for p in cpu_processes:
        p.join()

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Melee in emulator")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    parser.add_argument("--n_workers", type=int, help="Number of CPU workers")
    args = parser.parse_args()
    main(model_dir=args.model_dir, n_workers=args.n_workers)
