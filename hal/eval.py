import random
import signal
import sys
from pathlib import Path

import fire
import melee
import numpy as np
import torch

from melee import GameState, PlayerState, Button

from hal.io import get_default_melee_iso_path, get_default_dolphin_path


def load_model(model_arch: str, model_path: str, device: str) -> torch.nn.Module:
    pass


def infer_opponent_port(gamestate: GameState, ego_port: int):
    pass


def press_buttons(out_dict: dict[str, torch.Tensor], controller: melee.Controller, debug: bool = False):
    pass


def main(ego_port: int,
         ego_char: str,
         model_arch: str,
         model_path: str,
         device: str = "cpu",
         stage: str = "battlefield",
         dolphin_executable_path: str = get_default_dolphin_path(),
         iso_path: str = get_default_melee_iso_path(),
         address: str = "127.0.0.1",
         connect_code: str = "",
         debug: bool = False):
    log = None
    if debug:
        log = melee.Logger()

    ego_char = MAP_STR_TO_CHAR[ego_char.lower()]
    stage = MAP_STR_TO_STAGE[stage.lower()]
    costume = 0
    print("Loading model...")
    model = load_model(model_arch, model_path, device)

    console = melee.Console(path=dolphin_executable_path,
                            system="dolphin",
                            copy_home_directory=False,
                            slippi_address=address,
                            blocking_input=True,
                            logger=log)

    controller = melee.Controller(console=console,
                                  port=ego_port,
                                  type=melee.ControllerType.STANDARD)

    def signal_handler(sig, frame):
        console.stop()
        if debug:
            log.writelog()
            print("")  # because the ^C will be on the terminal
            print("Log file created: " + log.filename)
        print("Shutting down cleanly...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    console.run(iso_path=iso_path)
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    print("Console connected")

    # Plug our controller in
    #   Due to how named pipes work, this has to come AFTER running dolphin
    #   NOTE: If you're loading a movie file, don't connect the controller,
    #   dolphin will hang waiting for input and never receive it
    print("Connecting controller to console...")
    if not controller.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller connected")

    hidden_state = None

    while True:
        gamestate = console.step()
        if gamestate is None:
            continue

        if console.processingtime * 1000 > 12 or debug:
            print(f"Last frame took {console.processingtime * 1000} ms to process.")

        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            # todo: cache these ports
            if connect_code != "":
                ego_port = melee.gamestate.port_detector(gamestate, ego_char, costume)
            if ego_port > 0:
                opponent_port = infer_opponent_port(gamestate, ego_port=ego_port)
                ego: PlayerState = gamestate.players[ego_port]
                opponent: PlayerState = gamestate.players[opponent_port]

                frame = encode_frame(gamestate, ego, opponent)
                frame_arr = np.array(frame)[np.newaxis, ...]
                input_dict, _ = collate_and_preprocess_independent_axes([frame_arr], input_len=1, dataset_path='data/mang0')
                output_dict, hidden_state = model(input_dict, hidden_state)
                for k, v in output_dict.items():
                    output_dict[k] = torch.softmax(v, dim=-1)

                if output_dict is not None:
                    controller.release_all()
                    press_buttons(out_dict=output_dict, controller=controller, debug=debug)
                    controller.flush()
            else:
                # If the discovered port was unsure, reroll our costume for next time
                costume = random.randint(0, 4)
            if log:
                log.logframe(gamestate)
                log.writeframe()
        else:
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                controller,
                                                ego_char,
                                                stage,
                                                connect_code,
                                                costume=costume,
                                                autostart=False,
                                                swag=False)

            if log:
                log.skipframe()


if __name__ == "__main__":
    fire.Fire(main)
