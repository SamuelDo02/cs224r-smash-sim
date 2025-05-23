import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from glob import glob

class PKLReplayBuffer:
    """
    A replay buffer for pre-processed Melee pickle files
    """
    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size
        self.reset()

    def reset(self):
        """Reset the buffer"""
        self.observations = []  # Game state observations
        self.actions = []      # Player actions
        self.next_observations = []  # Next frame observations
        self.file_ids = []     # Track which file each frame came from
        self.current_size = 0

    def _create_observation(self, data: Dict, frame_idx: int) -> np.ndarray:
        """
        Create an observation vector from the frame data
        
        Args:
            data: Dictionary containing frame data
            frame_idx: Index of the frame to process
            
        Returns:
            Numpy array containing the observation
        """
        # Extract relevant features for the observation
        obs = np.array([
            data['p1_position_x'][frame_idx],
            data['p1_position_y'][frame_idx],
            data['p1_percent'][frame_idx],
            data['p1_facing'][frame_idx],
            data['p1_action'][frame_idx],
            data['p2_position_x'][frame_idx],
            data['p2_position_y'][frame_idx],
            data['p2_percent'][frame_idx],
            data['p2_facing'][frame_idx],
            data['p2_action'][frame_idx]
        ], dtype=np.float32)
        
        return obs

    def _create_action(self, data: Dict, frame_idx: int) -> np.ndarray:
        """
        Create an action vector from the frame data
        
        Args:
            data: Dictionary containing frame data
            frame_idx: Index of the frame to process
            
        Returns:
            Numpy array containing the action
        """
        # Extract all control inputs for player 1
        action = np.array([
            data['p1_main_stick_x'][frame_idx],    # Main stick X
            data['p1_main_stick_y'][frame_idx],    # Main stick Y
            data['p1_c_stick_x'][frame_idx],       # C-stick X
            data['p1_c_stick_y'][frame_idx],       # C-stick Y
            data['p1_l_shoulder'][frame_idx],      # L trigger
            data['p1_r_shoulder'][frame_idx],      # R trigger
            data['p1_button_a'][frame_idx],        # A button
            data['p1_button_b'][frame_idx],        # B button
            data['p1_button_x'][frame_idx],        # X button
            data['p1_button_y'][frame_idx],        # Y button
            data['p1_button_z'][frame_idx],        # Z button
            data['p1_button_start'][frame_idx]     # Start button
        ], dtype=np.float32)
        
        return action

    def add_pkl_file(self, pkl_path: str) -> None:
        """
        Add a pickle file to the buffer
        
        Args:
            pkl_path: Path to the .pkl file
        """
        print(f"Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Get number of frames in this file
        num_frames = len(data['frame'])
        
        # Add frames to buffer, excluding the last frame since we need next_observation
        for i in range(num_frames - 1):
            if self.current_size >= self.max_size:
                return
                
            # Create observation and action vectors
            observation = self._create_observation(data, i)
            action = self._create_action(data, i)
            next_observation = self._create_observation(data, i + 1)
            
            # Add to buffer
            self.observations.append(observation)
            self.actions.append(action)
            self.next_observations.append(next_observation)
            self.file_ids.append(Path(pkl_path).stem)
            self.current_size += 1

    def add_directory(self, directory: str) -> None:
        """
        Add all pickle files from a directory
        
        Args:
            directory: Path to directory containing .pkl files
        """
        pkl_files = sorted(glob(str(Path(directory) / "*.pkl")))
        for pkl_file in pkl_files:
            self.add_pkl_file(pkl_file)
        print(f"Loaded {self.current_size} total frames")

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (observations, actions)
        """
        if self.current_size == 0:
            raise ValueError("Buffer is empty!")
            
        indices = np.random.randint(0, self.current_size, size=batch_size)
        
        return (
            np.stack([self.observations[i] for i in indices]),
            np.stack([self.actions[i] for i in indices])
        )

    def save(self, save_path: str) -> None:
        """Save the replay buffer to a file"""
        np.savez(
            save_path,
            observations=np.stack(self.observations),
            actions=np.stack(self.actions),
            next_observations=np.stack(self.next_observations),
            file_ids=np.array(self.file_ids)
        )

    def load(self, load_path: str) -> None:
        """Load the replay buffer from a file"""
        data = np.load(load_path)
        self.observations = list(data['observations'])
        self.actions = list(data['actions'])
        self.next_observations = list(data['next_observations'])
        self.file_ids = list(data['file_ids'])
        self.current_size = len(self.observations) 