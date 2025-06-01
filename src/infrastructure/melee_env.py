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

from infrastructure.menu import MatchupMenuHelper
from infrastructure.controller import Controller

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

        self.action_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(12,),  # [main_x, main_y, c_x, c_y, l_trigger, r_trigger, a, b, x, y, z, start]
            dtype=np.float32
        )

        # Define observation space for single frame
        # [p1_x, p1_y, p1_percent, p1_facing, p1_action_state,
        #  p2_x, p2_y, p2_percent, p2_facing, p2_action_state]
        self.observation_space = spaces.Box(
            low=np.array([-250, -250, 0, -1, 0, -250, -250, 0, -1, 0], dtype=np.float32),
            high=np.array([250, 250, 999, 1, 386, 250, 250, 999, 1, 386], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize frame buffer for observation history
        self.frame_buffer = [
            np.zeros(10, dtype=np.float32) for _ in range(5)
        ]
        self.frame_window = 5

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
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)

        # Initialize state variables
        self.frame_count = 0
        self.match_started = False
        self.game_ended = False
        self.max_frames = 10800  # 3 minutes at 60fps
        self.gamestate = None
        self.num_stocks = 3
        self.p1_reward = self.num_stocks * 400
        self.p2_reward = self.num_stocks * 400

        self.start_game()

    def start_game(self):
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
        if hasattr(self, 'console'):
            self.console.stop()
        if hasattr(self, 'log'):
            self.log.writelog()
            logger.info(f"Log file created: {self.log.filename}")

    def gaming(self):
        """Return True if the game is still running"""
        return self.frame_count < self.max_frames and self.match_started and not self.game_ended

    def get_gamestate(self):
        """Get the current gamestate"""
        return self.gamestate
    
    def get_observation(self):
        """Get the current observation from the gamestate"""
        if self.gamestate is None or 1 not in self.gamestate.players or 2 not in self.gamestate.players:
            # If gamestate or players aren't loaded yet, return zeros
            return np.zeros(10, dtype=np.float32)
            
        p1 = self.gamestate.players[1]
        p2 = self.gamestate.players[2]
        
        obs = np.array([
            p1.position.x, p1.position.y, p1.percent, p1.facing, p1.action.value,
            p2.position.x, p2.position.y, p2.percent, p2.facing, p2.action.value
        ], dtype=np.float32)
        
        return obs

    def get_frame_history(self):
        """Get the observation history of past 5 frames"""
        # If we don't have enough frames yet, pad with zeros
        while len(self.frame_buffer) < self.frame_window:
            self.frame_buffer.insert(0, np.zeros(10, dtype=np.float32))
            
        # Get current observation
        current_obs = self.get_observation()
        
        # Update frame buffer
        self.frame_buffer.append(current_obs)
        if len(self.frame_buffer) > self.frame_window:
            self.frame_buffer.pop(0)
            
        # Stack frames
        return np.concatenate(self.frame_buffer)

    def reset(self, seed=None):
        """Reset the environment to start a new episode"""
        # Reset game state variables
        self.frame_count = 0
        self.match_started = False
        self.game_ended = False
        self.frame_buffer = []  # Clear frame buffer
        
        # Start new game
        print("Starting new game...")
        self.console.run(iso_path=str(ISO_PATH))
        
        # Connect to console
        if not self.console.connect():
            print("ERROR: Failed to connect to the console.")
            sys.exit(-1)
            
        # Connect controllers
        if not self.controller1.connect():
            print("ERROR: Failed to connect controller 1.")
            sys.exit(-1)
        if not self.controller2.connect():
            print("ERROR: Failed to connect controller 2.")
            sys.exit(-1)
            
        # Get initial gamestate
        self.gamestate = self.console.step()
        
        # Navigate through menus until in game
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
        
        # Return initial observation with frame history
        return self.get_frame_history()

    def step(self, action):
        """Execute one timestep in the environment"""
        # Take a step in the environment
        self.gamestate = self.console.step()
        if self.gamestate is None:
            return None, 0, True, {"issue": "gamestate is None"}
                
        self.frame_count += 1
            
        # Bot control for player 1
        self.controller.act(action)
            
        # Log the frame
        self.log.logframe(self.gamestate)
        self.log.writeframe()
            
        # Get observation with frame history
        obs = self.get_frame_history()
            
        # Check if game ended (someone lost all stocks)
        if 1 in self.gamestate.players and 2 in self.gamestate.players:
            p1 = self.gamestate.players[1]
            p2 = self.gamestate.players[2]
            
            # Check if game ended (someone lost all stocks)
            if p1.stock == 0 or p2.stock == 0:
                logger.info(f"Game ended! P1 stocks: {p1.stock}, P2 stocks: {p2.stock}")
                self.game_ended = True

                # Write the log file
                self.log.writelog()
                print(f"Replay saved to: {self.log.filename}")
                
                # Copy replay to a more obvious location
                replay_file = Path(self.log.filename)
                if replay_file.exists():
                    new_replay_path = REPLAY_DIR / f"bot_vs_cpu_{replay_file.name}"
                    replay_file.rename(new_replay_path)
                    print(f"Replay moved to: {new_replay_path}")

                return obs, 0, True, {"issue": "game ended"}
            
            # Progress indicator
            if self.frame_count % 300 == 0:  # Every 5 seconds
                logger.info(f"Frame {self.frame_count} ({self.frame_count/60:.1f}s) - P1: {p1.stock} stocks, {p1.percent:.1f}%, P2: {p2.stock} stocks, {p2.percent:.1f}%")

            p1_reward = (int(p1.stock) * 400 - int(p1.percent)) if p1.stock > 0 else -10000
            p2_reward = (int(p2.stock) * 400 - int(p2.percent)) if p2.stock > 0 else -10000
            last_reward = self.p1_reward - self.p2_reward
            reward = (p1_reward - p2_reward) - last_reward
            self.p1_reward = p1_reward
            self.p2_reward = p2_reward
            return obs, reward, False, {"issue": "game in progress"}

        return obs, 0, False, {"issue": "game not started"}

    def render(self):
        """Render the environment"""
        pass