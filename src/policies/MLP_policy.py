"""
Defines a pytorch policy for Melee behavior cloning.
"""

import torch
from torch import nn
from torch import distributions
import numpy as np

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: str = 'tanh',
        output_activation: str = 'identity',
) -> nn.Module:
    """
    Builds a feedforward neural network
    
    Arguments:
        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer
        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer
        
    Returns:
        MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    
    # First layer
    layers.append(nn.Linear(input_size, size))
    layers.append(activation)
    
    # Hidden layers
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(size, size))
        layers.append(activation)
    
    # Output layer
    layers.append(nn.Linear(size, output_size))
    layers.append(output_activation)

    return nn.Sequential(*layers)

class MLPPolicySL(nn.Module):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 **kwargs
                 ):
        super().__init__()

        # Initialize variables
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create networks
        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.mean_net.to(self.device)
        
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=self.device)
        )
        self.logstd.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.mean_net.parameters()) + [self.logstd],
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
        mean = self.mean_net(observation)
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
        log_probs = dist.log_prob(actions)
        loss = -log_probs.sum(dim=-1).mean()
    
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            'Training Loss': loss.item(),
        }

