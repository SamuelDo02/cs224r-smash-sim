"""Adapted from Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT."""
import math
from typing import Any

import attr
from emulator.constants import JOYSTICK_POINTS, NUM_JOYSTICK_POSITIONS
import numpy as np
from policies.metrics import compute_action_accuracy, compute_success_rate
from policies.preprocessor import Preprocessor
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn import functional as F
from torch.optim import Adam


@attr.s(auto_attribs=True, frozen=True)
class GPTConfig:
    block_size: int
    n_embd: int 
    n_layer: int
    n_head: int
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    learning_rate: float = 1e-3  # Default learning rate
    multi_token_heads: tuple[int, ...] = (1,)  # Numbers of frames in the future to predict

class CausalSelfAttentionRelativePosition(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor):
        B, L, D = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = v.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal self-attention; Self-attend: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :L, :L] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class BlockRelativePosition(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionRelativePosition(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTARPolicy(nn.Module):
    """GPT policy with relative positional embeddings and autoregressive MLP output heads."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        gpt_config: GPTConfig = GPTConfig(block_size=1024, n_embd=512, n_layer=6, n_head=8, dropout=0.2),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.gpt_config = gpt_config
        self.block_size = gpt_config.block_size
        self.multi_token_heads = gpt_config.multi_token_heads
        self.num_multi_token_output_heads = len(self.multi_token_heads)

        # Initialize optimizer

        # Numeric + embedded feature sizes defined programmatically in InputPreprocessConfig
        self.input_size = self.preprocessor.input_size  # G
        self.n_embd = gpt_config.n_embd  # D

        # Categorical input embeddings
        self.emb_config = self.preprocessor.data_config
        self.stage_emb = nn.Embedding(self.emb_config.num_stages, self.emb_config.stage_embedding_dim)
        self.character_emb = nn.Embedding(self.emb_config.num_characters, self.emb_config.character_embedding_dim)
        self.action_emb = nn.Embedding(self.emb_config.num_actions, self.emb_config.action_embedding_dim)

        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, gpt_config.n_embd),  # G -> D
                drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList([BlockRelativePosition(gpt_config) for _ in range(gpt_config.n_layer)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )

        # Output heads
        self.target_shapes_by_head = self.preprocessor.target_config.target_shapes_by_head
        shoulder_output_dim = self.target_shapes_by_head["shoulder"][0]
        c_stick_output_dim = self.target_shapes_by_head["c_stick"][0]
        main_stick_output_dim = self.target_shapes_by_head["main_stick"][0]
        button_output_dim = self.target_shapes_by_head["buttons"][0]

        # Update output dimensions for one-hot predictions
        self.num_joystick_positions = len(JOYSTICK_POINTS)
        
        # Multiply output sizes by number of future frames to predict
        shoulder_output_size = shoulder_output_dim * self.num_multi_token_output_heads
        c_stick_output_size = self.num_joystick_positions * self.num_multi_token_output_heads  # One-hot for C-stick
        main_stick_output_size = self.num_joystick_positions * self.num_multi_token_output_heads  # One-hot for main stick
        button_output_size = button_output_dim * self.num_multi_token_output_heads

        # Input sizes for each head
        shoulder_input_size = self.n_embd
        c_stick_input_size = self.n_embd + shoulder_output_size
        main_stick_input_size = self.n_embd + shoulder_output_size + c_stick_output_size
        button_input_size = self.n_embd + shoulder_output_size + c_stick_output_size + main_stick_output_size

        # Update output heads with softmax for one-hot predictions
        self.shoulder_head = nn.Sequential(
            nn.LayerNorm(shoulder_input_size, bias=gpt_config.bias),
            nn.Linear(shoulder_input_size, shoulder_input_size // 2),
            nn.GELU(),
            nn.Linear(shoulder_input_size // 2, shoulder_output_size),
        )

        self.c_stick_head = nn.Sequential(
            nn.LayerNorm(c_stick_input_size, bias=gpt_config.bias),
            nn.Linear(c_stick_input_size, c_stick_input_size // 2),
            nn.GELU(),
            nn.Linear(c_stick_input_size // 2, c_stick_output_size),
            nn.Softmax(dim=-1)  # Add softmax for one-hot prediction
        )

        self.main_stick_head = nn.Sequential(
            nn.LayerNorm(main_stick_input_size, bias=gpt_config.bias),
            nn.Linear(main_stick_input_size, main_stick_input_size // 2),
            nn.GELU(),
            nn.Linear(main_stick_input_size // 2, main_stick_output_size),
            nn.Softmax(dim=-1)  # Add softmax for one-hot prediction
        )

        self.button_head = nn.Sequential(
            nn.LayerNorm(button_input_size, bias=gpt_config.bias),
            nn.Linear(button_input_size, button_input_size // 2),
            nn.GELU(),
            nn.Linear(button_input_size // 2, button_output_size),
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

        self.optimizer = Adam(self.parameters(), lr=gpt_config.learning_rate)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opp_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opp_action"]).squeeze(-2),
                inputs["gamestate"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict) -> TensorDict:
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # Concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        # Initialize result dictionary
        result = {}
        
        # For each future frame we want to predict
        for offset in self.multi_token_heads:
            # Get the current sequence up to this point
            current_x = x_BLD
            
            # Process all time steps at once for each output mode, autoregressively decode next head
            shoulder = self.shoulder_head(current_x)
            c_stick = self.c_stick_head(torch.cat((current_x, shoulder.detach()), dim=-1))
            main_stick = self.main_stick_head(torch.cat((current_x, shoulder.detach(), c_stick.detach()), dim=-1))
            button = self.button_head(torch.cat((current_x, shoulder.detach(), c_stick.detach(), main_stick.detach()), dim=-1))

            # Store predictions for this frame
            result[f"shoulder_{offset}"] = shoulder[:, :, :]  # Take last timestep
            result[f"c_stick_{offset}"] = c_stick[:, :, :]
            result[f"main_stick_{offset}"] = main_stick[:, :, :]
            result[f"buttons_{offset}"] = button[:, :, :]

            # If we're not at the last frame, update the input sequence with our predictions
            if offset < self.multi_token_heads[-1]:
                # Create new input for next frame by appending current predictions
                new_input = torch.cat([
                    inputs["gamestate"],
                    shoulder[:, :, -1, :],
                    c_stick[:, :, -1, :],
                    main_stick[:, :, -1, :],
                    button[:, :, -1, :]
                ], dim=-1)
                
                # Update the input sequence
                inputs["gamestate"] = new_input
                combined_inputs_BLG = self._embed_inputs(inputs)
                proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)
                x_BLD = self.transformer.drop(proj_inputs_BLD)
                for block in self.transformer.h:
                    x_BLD = block(x_BLD)
                x_BLD = self.transformer.ln_f(x_BLD)

        return TensorDict(result, batch_size=(B, L))
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from the policy."""
        inputs = self.preprocessor.preprocess_observation(obs)
        outputs = self.forward(inputs)
        
        # map this to the action space 
        # [main_x, main_y, c_x, c_y, l_trigger, r_trigger, a, b, x, y, z, start]
        action = np.zeros(12)
        
        # Get the predicted one-hot vectors for the joysticks
        main_stick_onehot = outputs['main_stick_1'][0, -1, :NUM_JOYSTICK_POSITIONS]
        c_stick_onehot = outputs['c_stick_1'][0, -1, :NUM_JOYSTICK_POSITIONS]
        
        # Convert one-hot predictions to indices
        main_stick_idx = torch.argmax(main_stick_onehot).item()
        c_stick_idx = torch.argmax(c_stick_onehot).item()
        
        # Get coordinates from predefined points
        main_x, main_y = JOYSTICK_POINTS[main_stick_idx]
        c_x, c_y = JOYSTICK_POINTS[c_stick_idx]
        
        # Fill in the action array
        action[0] = main_x
        action[1] = main_y
        action[2] = c_x
        action[3] = c_y
        action[4] = outputs['shoulder_1'][0, -1, 0].item()  # L trigger
        action[5] = outputs['shoulder_1'][0, -1, 1].item()  # R trigger
        action[6] = outputs['buttons_1'][0, -1, 0].item()   # A
        action[7] = outputs['buttons_1'][0, -1, 1].item()   # B
        action[8] = outputs['buttons_1'][0, -1, 2].item()   # X
        action[9] = outputs['buttons_1'][0, -1, 3].item()   # Y
        action[10] = outputs['buttons_1'][0, -1, 4].item()  # Z
        action[11] = outputs['buttons_1'][0, -1, 5].item()  # Start
        
        return action
    

    def update(self, observations, actions, train=True):
        """
        Update the policy using behavior cloning
        
        Args:
            observations: Batch of observations (numpy array or TensorDict)
            actions: Batch of actions (numpy array or TensorDict)
            train: Whether to update the policy
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
            
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(0)
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)

        # Preprocess observations and actions
        observations = self.preprocessor.preprocess_observation(observations)
        actions = self.preprocessor.preprocess_action(actions)

        # Get action distribution
        outputs = self.forward(observations)

        # Compute loss for each component and future frame
        loss = 0.0
        loss_dict = {}
        
        # Get the base component names without the frame offset
        base_components = ["buttons", "main_stick", "c_stick", "shoulder"]

        # print(actions["main_stick"])
        
        for component in base_components:
            component_loss = 0.0
            # For each future frame we're predicting
            for offset in self.multi_token_heads:
                output_key = f"{component}_{offset}"
                # Only compute loss for frames we have ground truth for
                if offset <= actions[component].shape[1]:
                    if component in ["main_stick", "c_stick"]:
                        # print(actions[component])
                        # action component is the one hot encoded joystick point
                        target_indices = actions[component][0].argmax(dim=-1)   
                        
                            
                        # Use cross entropy loss for one-hot predictions
                        frame_loss = F.cross_entropy(
                            outputs[output_key][0].view(-1, len(JOYSTICK_POINTS)),
                            target_indices.view(-1)
                        )
                    else:
                        # Use MSE loss for other components
                        frame_loss = F.mse_loss(outputs[output_key][0], actions[component][0])
                    
                    component_loss += frame_loss
                    loss_dict[f'{output_key}_loss'] = frame_loss.item()
            
            loss += component_loss
            loss_dict[f'{component}_total_loss'] = component_loss.item()

        if train:
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_dict['Training Loss'] = loss.item()

        # Convert outputs to flat tensor for metrics (using only the first predicted frame)
        first_frame = self.multi_token_heads[0]
        pred_flat = torch.cat([
            outputs[f'main_stick_{first_frame}'],
            outputs[f'c_stick_{first_frame}'], 
            outputs[f'shoulder_{first_frame}'],
            outputs[f'buttons_{first_frame}']
        ], dim=-1).squeeze(0)

        # Convert actions to flat tensor
        actions_flat = torch.cat([
            actions['main_stick'],
            actions['c_stick'],
            actions['shoulder'], 
            actions['buttons']
        ], dim=-1).squeeze(0)

        # Compute accuracy metrics
        thresholds = [0.05, 0.1, 0.2]  # Thresholds for accuracy computation
        accuracy_metrics = compute_action_accuracy(pred_flat, actions_flat, thresholds)
        success_rate = compute_success_rate(pred_flat, actions_flat)

        loss_dict.update({
            'Training Loss': loss.item(),
            'Success Rate': success_rate,
        })
        loss_dict.update(accuracy_metrics)
        return loss_dict
    
    def save(self, filepath):
        """
        Save the policy to a file
        """
        torch.save(self.state_dict(), filepath)


# Arch.register(
#     "GPTv5Controller-512-6-8-dropout",
#     GPTv5Controller,
#     gpt_config=GPTConfig(block_size=1024, n_embd=512, n_layer=6, n_head=8, dropout=0.2),
# )
