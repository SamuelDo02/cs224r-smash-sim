"""
Defines a transformer policy for Melee behavior cloning.
"""

import torch
from torch import nn
from torch import distributions
import numpy as np

class TransformerPolicySL(nn.Module):
    """
    Defines a Transformer for supervised learning which maps observations to actions

    Attributes
    ----------
    transformer: nn.TransformerEncoder
        A transformer encoder that processes the input sequence
    input_proj: nn.Linear
        Projects input observations to transformer dimension
    output_proj: nn.Linear
        Projects transformer output to action dimension
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 n_heads=4,
                 dropout=0.1,
                 learning_rate=1e-4,
                 **kwargs
                 ):
        super().__init__()

        # Initialize variables
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.n_heads = n_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Set device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Input projection
        self.input_proj = nn.Linear(ob_dim, size)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, size))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=size,
            nhead=n_heads,
            dim_feedforward=size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            *[nn.Linear(size, size) for _ in range(n_layers-1)],
            nn.Linear(size, ac_dim)
        )
        
        # Log standard deviation
        self.logstd = nn.Parameter(
            torch.zeros(ac_dim, dtype=torch.float32, device=self.device)
        )
        
        # Move to device
        self.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        Save the policy to a file
        """
        torch.save(self.state_dict(), filepath)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get an action for the given observation
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # Convert to tensor and get action
        observation = torch.as_tensor(observation, dtype=torch.float32).to(self.device)
        dist = self.forward(observation)
        action = dist.sample()
        
        return action.cpu().detach().numpy()

    def forward(self, observation: torch.FloatTensor) -> distributions.Distribution:
        """
        Forward pass through the network
        """
        # Add sequence dimension if needed
        if len(observation.shape) == 2:
            observation = observation.unsqueeze(1)
            
        # Project input
        x = self.input_proj(observation)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project to action space
        mean = self.output_proj(x).squeeze(1)  # Remove sequence dimension
        
        # Get standard deviation
        std = torch.exp(self.logstd)
        
        return distributions.Normal(mean, std)

    def update(self, observations, actions, train=True):
        """
        Update the policy using behavior cloning
        
        Args:
            observations: Batch of observations
            actions: Batch of actions to imitate
            train: Whether to update the network or just compute loss
        """
        # Convert numpy arrays to tensors
        observations = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)

        # Get action distribution and compute loss
        dist = self.forward(observations)
        # print(actions, dist)
        log_probs = dist.log_prob(actions)
        loss = -log_probs.sum(dim=-1).mean()
    
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            'Training Loss': loss.item(),
        }
