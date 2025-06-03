"""Preprocessor for GPT policy that handles data formatting and embedding configuration."""
from dataclasses import dataclass
from typing import Dict, Tuple

from emulator.constants import NUM_JOYSTICK_POSITIONS, get_closest_joystick_point
import torch
from tensordict import TensorDict
from melee import Action
from melee import Character
from melee import Stage


@dataclass
class EmbeddingConfig:
    """Configuration for categorical embeddings."""
    num_stages: int = len(Stage)  # Number of stages in Melee
    num_characters: int = len(Character)  # Number of playable characters in Melee
    num_actions: int = len(Action)  # Number of possible actions
    
    stage_embedding_dim: int = 4
    character_embedding_dim: int = 12
    action_embedding_dim: int = 32


@dataclass
class TargetConfig:
    """Configuration for output target shapes."""
    target_shapes_by_head: Dict[str, Tuple[int, ...]] = None
    
    def __post_init__(self):
        if self.target_shapes_by_head is None:
            # Default shapes for Melee controller outputs
            self.target_shapes_by_head = {
                "shoulder": (2,),  # L and R shoulder buttons
                "c_stick": (2,),   # C-stick X and Y
                "main_stick": (2,), # Main stick X and Y
                "buttons": (6,)     # A, B, X, Y, Z, Start
            }


class Preprocessor:
    """Preprocessor for GPT policy that handles data formatting and embedding configuration."""
    
    def __init__(
        self,
        frame_window: int = 5,
        embedding_config: EmbeddingConfig = None,
        target_config: TargetConfig = None,
    ):
        self.frame_window = frame_window
        self._embedding_config = embedding_config or EmbeddingConfig()
        self._target_config = target_config or TargetConfig()
        
        # Calculate input size based on embeddings and numerical features
        self.input_size = (
            self._embedding_config.stage_embedding_dim +
            self._embedding_config.character_embedding_dim * 2 +  # ego and opponent
            self._embedding_config.action_embedding_dim * 2 +    # ego and opponent
            50   # gamestate features (position, percent, facing, action state)
        )
    @property
    def data_config(self) -> EmbeddingConfig:
        """Get the embedding configuration."""
        return self._embedding_config
    
    @property
    def target_config(self) -> TargetConfig:
        """Get the target configuration."""
        return self._target_config
    
    def preprocess_observation(self, obs: torch.Tensor) -> TensorDict:
        """Convert raw observation tensor to TensorDict format."""
        B, L, _ = obs.shape
        
        # Split observation into components based on MeleeEnv observation space:
        # [p1_x, p1_y, p1_percent, p1_facing, p1_action_state,
        #  p2_x, p2_y, p2_percent, p2_facing, p2_action_state]
        
        # Each player has 5 features
        p1_start = 0
        p2_start = 5
        
        # Ensure action indices are within valid range
        ego_action = obs[:, :, p1_start+4:p1_start+5].long()
        opp_action = obs[:, :, p2_start+4:p2_start+5].long()
        
        # Clamp action indices to valid range
        ego_action = torch.clamp(ego_action, 0, self._embedding_config.num_actions - 1)
        opp_action = torch.clamp(opp_action, 0, self._embedding_config.num_actions - 1)
        
        return TensorDict(
            {
                "gamestate": obs[:, :, :],  # Pass through all features
                "ego_pos": obs[:, :, p1_start:p1_start+2],  # x,y
                "ego_percent": obs[:, :, p1_start+2:p1_start+3],
                "ego_facing": obs[:, :, p1_start+3:p1_start+4],
                "ego_action": ego_action,
                "opp_pos": obs[:, :, p2_start:p2_start+2],  # x,y 
                "opp_percent": obs[:, :, p2_start+2:p2_start+3],
                "opp_facing": obs[:, :, p2_start+3:p2_start+4],
                "opp_action": opp_action,
                "stage": torch.zeros((B, L, 1), dtype=torch.long, device=obs.device),  # Always stage 0
                "ego_character": torch.full((B, L, 1), 0x0a, dtype=torch.long, device=obs.device),  # Always Fox (0x0a)
                "opp_character": torch.full((B, L, 1), 0x0a, dtype=torch.long, device=obs.device),  # Always Fox (0x0a)
            },
            batch_size=(B, L),
        )
    
    def preprocess_action(self, action: torch.Tensor) -> TensorDict:
        """Convert raw action tensor to TensorDict format."""
        B, L, _ = action.shape
        
        # Split action into components based on MeleeEnv action space:
        # [main_x, main_y, c_x, c_y, l_trigger, r_trigger, a, b, x, y, z, start]
        
        # Convert continuous joystick values to one-hot tensors
        main_stick_onehot = torch.zeros((B, L, NUM_JOYSTICK_POSITIONS), device=action.device)
        c_stick_onehot = torch.zeros((B, L, NUM_JOYSTICK_POSITIONS), device=action.device)
        
        # Process each batch and timestep
        for b in range(B):
            for l in range(L):
                # Get main stick coordinates
                main_x = action[b, l, 0].item()
                main_y = action[b, l, 1].item()
                main_stick_onehot[b, l] = get_closest_joystick_point(main_x, main_y)
                
                # Get C-stick coordinates
                c_x = action[b, l, 2].item()
                c_y = action[b, l, 3].item()
                c_stick_onehot[b, l] = get_closest_joystick_point(c_x, c_y)
        
        return TensorDict(
            {
                "main_stick": main_stick_onehot,  # One-hot encoded main stick
                "c_stick": c_stick_onehot,        # One-hot encoded C-stick
                "shoulder": action[:, :, 4:6],    # l_trigger, r_trigger
                "buttons": action[:, :, 6:12],    # a, b, x, y, z, start
            },
            batch_size=(B, L)
        )