import gym
from gym import spaces

import os
import sys
import signal
import melee
import melee.enums as enums
import numpy as np
from pathlib import Path
from loguru import logger

from menu import MatchupMenuHelper
from controller import Controller

GIT_ROOT = Path(__file__).parent.parent.parent
EMULATOR_PATH = GIT_ROOT / "squashfs-root" / "usr" / "bin" / "dolphin-emu"
ISO_PATH = GIT_ROOT / "melee.iso"
REPLAY_DIR = GIT_ROOT / "replays"

def get_gui_console_kwargs():
    """Get console kwargs for GUI-enabled emulator (modified from hal project)."""
    REPLAY_DIR.mkdir(exist_ok=True, parents=True)
    console_kwargs = {
        "path": str(EMULATOR_PATH),
        "is_dolphin": True,
        "tmp_home_directory": True,
        "copy_home_directory": False,
        "replay_dir": str(REPLAY_DIR),
        "blocking_input": False,
        "slippi_port": 51441,  # must use default port for local mainline/Ishiiruka
        "online_delay": 0,  # 0 frame delay for local evaluation
        "logger": None,
        "setup_gecko_codes": True,
        "fullscreen": False,
        "gfx_backend": "",  # Use default backend instead of Null
        "disable_audio": True,  # Still disable audio for headless
        "use_exi_inputs": False,
        "enable_ffw": False,
    }
    return console_kwargs

class MeleeEnv(gym.Env):
    """
    A gym environment wrapper for Super Smash Bros. Melee
    """
    def __init__(self):
        super().__init__()

        print("Starting Slippi Online emulator with bot vs CPU match...")

        # Set up logging for the match
        self.log = melee.Logger()

        # Create console with GUI settings but still headless-friendly
        console_kwargs = get_gui_console_kwargs()
        print(f"Console kwargs: {console_kwargs}")

        self.console = melee.Console(**console_kwargs)

        # Create controllers
        self.controller1 = melee.Controller(console=self.console, port=1, type=melee.ControllerType.STANDARD)
        self.controller2 = melee.Controller(console=self.console, port=2, type=melee.ControllerType.STANDARD)

        # Create controller for bot
        self.controller = Controller(self.controller1)

        # Create menu helper
        self.menu_helper = MatchupMenuHelper(
            controller_1=self.controller1,
            controller_2=self.controller2,
            character_1=melee.Character.FOX,
            character_2=melee.Character.FALCO,
            stage=melee.Stage.BATTLEFIELD,
            opponent_cpu_level=5  # Level 5 CPU
        )

        # Set up environment variables for software rendering
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["MESA_GL_VERSION_OVERRIDE"] = "2.1"

        # Signal handler for clean shutdown
        def signal_handler(sig, frame):
            self.console.stop()
            if self.log:
                self.log.writelog()
                logger.info(f"Log file created: {self.log.filename}")
            logger.info("Shutting down cleanly...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)

        try:
            print("Starting emulator...")
            self.console.run(iso_path=str(ISO_PATH))
        
            print("Connecting to console...")
            if not self.console.connect():
                print("ERROR: Failed to connect to the console.")
                sys.exit(-1)
            print("Console connected")
        
            print("Connecting controllers...")
            if not self.controller1.connect():
                print("ERROR: Failed to connect controller 1.")
                sys.exit(-1)
            print("Controller 1 connected")
        
            if not self.controller2.connect():
                print("ERROR: Failed to connect controller 2.")
                sys.exit(-1)
            print("Controller 2 connected")
        
            # Main game loop
            print("Starting game loop...")
            self.frame_count = 0
            self.match_started = False
            self.game_ended = False
            self.max_frames = 10800  # 3 minutes at 60fps

            self.gamestate = self.console.step()
            # Advance past menus
            while self.gamestate.menu_state not in [enums.Menu.IN_GAME, enums.Menu.SUDDEN_DEATH]:
                # Navigate through menus
                self.menu_helper.select_character_and_stage(self.gamestate)
                # Skip logging menu frames
                self.log.skipframe() 
                # Increment frame count
                self.frame_count += 1

                # Emergency break if stuck in menus too long
                if self.frame_count > 1800:  # 30 seconds in menus
                    logger.warning("Stuck in menus for too long, exiting...")
                    sys.exit(-1)

                self.gamestate = self.console.step()

            # Match started!
            logger.info("Match started!")
            self.match_started = True

        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Clean up resources"""
        self.console.stop()
        if self.log:
            self.log.writelog()
            logger.info(f"Log file created: {self.log.filename}")

    def gaming(self):
        """Return True if the game is still running"""
        return self.frame_count < self.max_frames and self.match_started and not self.game_ended

    def get_gamestate(self):
        """Get the current gamestate"""
        return self.gamestate

    def reset(self, seed=None):
        """Reset the environment to start a new episode"""
        pass

    def step(self, action):
        """Execute one timestep in the environment"""
        # For behavior cloning from pre-processed data, we don't need actual environment steps
        self.gamestate = self.console.step()
        if self.gamestate is None:
            return None
                
        self.frame_count += 1
            
        # Bot control for player 1
        self.controller.act(action)
            
        # Log the frame
        self.log.logframe(self.gamestate)
        self.log.writeframe()
            
        # Check if game ended (someone lost all stocks)
        if 1 in self.gamestate.players and 2 in self.gamestate.players:
            p1_stocks = self.gamestate.players[1].stock
            p2_stocks = self.gamestate.players[2].stock
            
            if p1_stocks == 0 or p2_stocks == 0:
                logger.info(f"Game ended! P1 stocks: {p1_stocks}, P2 stocks: {p2_stocks}")
                self.game_ended = True

                # Write the log file
                self.log.writelog()
                print(f"Replay saved to: {self.log.filename}")
                
                # Copy replay to a more obvious location
                import shutil
                replay_file = Path(self.log.filename)
                if replay_file.exists():
                    new_replay_path = REPLAY_DIR / f"bot_vs_cpu_{replay_file.name}"
                    shutil.copy2(replay_file, new_replay_path)
                    print(f"Replay also copied to: {new_replay_path}")

                return None
            
        # Progress indicator
        if self.frame_count % 300 == 0:  # Every 5 seconds
            p1_info = f"P1: {self.gamestate.players[1].stock} stocks, {self.gamestate.players[1].percent:.1f}%" if 1 in self.gamestate.players else "P1: N/A"
            p2_info = f"P2: {self.gamestate.players[2].stock} stocks, {self.gamestate.players[2].percent:.1f}%" if 2 in self.gamestate.players else "P2: N/A"
            logger.info(f"Frame {self.frame_count} ({self.frame_count/60:.1f}s) - {p1_info}, {p2_info}")

    def render(self):
        """Render the environment"""
        pass 