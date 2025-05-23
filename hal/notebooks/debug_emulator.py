import sys

import melee
from loguru import logger

from hal.emulator_helper import console_manager
from hal.emulator_helper import get_headless_console_kwargs
from hal.emulator_helper import self_play_menu_helper
from hal.local_paths import ISO_PATH

PLAYER_1_PORT = 1
PLAYER_2_PORT = 2


def run_episode(rank: int, port: int, max_steps: int = 8 * 60 * 60) -> None:
    console_kwargs = get_headless_console_kwargs(rank=rank, udp_port=port)
    console = melee.Console(**console_kwargs)
    logger.info(f"Worker {rank}: slippi address {console.slippi_address}, port {console.slippi_port}")

    controller_1 = melee.Controller(console=console, port=PLAYER_1_PORT, type=melee.ControllerType.STANDARD)
    controller_2 = melee.Controller(console=console, port=PLAYER_2_PORT, type=melee.ControllerType.STANDARD)

    # Run the console
    console.run(iso_path=ISO_PATH)
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
        while i < max_steps:
            gamestate = console.step()
            if gamestate is None:
                logger.info("Gamestate is None")
                break
            logger.info(f"Iteration {i}: Menu state: {gamestate.menu_state}")

            if console.processingtime * 1000 > 12:
                logger.info("WARNING: Last frame took " + str(console.processingtime * 1000) + "ms to process.")

            if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                logger.info("Menu helper")
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
                    logger.info("Match started")

                i += 1


# if __name__ == "__main__":
#     cpu_processes = []
#     ports = find_open_udp_ports(2)
#     for i, port in enumerate(ports):
#         p: mp.Process = mp.Process(
#             target=run_episode_wrapper,
#             kwargs=dict(rank=i, port=port),
#         )
#         p.start()
#         cpu_processes.append(p)

#     for p in cpu_processes:
#         p.join()

if __name__ == "__main__":
    # ports = find_open_udp_ports(1)
    run_episode(rank=0, port=51441)
