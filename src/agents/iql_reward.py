import torch
import torch.nn.functional as F

class IQLReward:
    """
    Reward function class for Implicit Q-Learning that handles value and Q-function losses
    """
    def __init__(self, temperature=10.0, expectile=0.9995, max_grad_norm=1.0):
        self.temperature = temperature
        self.expectile = expectile
        self.max_grad_norm = max_grad_norm

    def compute_value_loss(self, q_values, value_pred):
        """
        Compute expectile loss for value function
        
        Args:
            q_values: Q-function predictions
            value_pred: Value function predictions
            
        Returns:
            torch.Tensor: Expectile loss value
        """
        diff = q_values - value_pred
        # Clip differences to prevent extreme values
        diff = torch.clamp(diff, min=-10.0, max=10.0)
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * (diff ** 2)).mean()

    def compute_q_loss(self, q_pred, value_pred):
        """
        Compute MSE loss for Q-function
        
        Args:
            q_pred: Q-function predictions
            value_pred: Value function predictions
            
        Returns:
            torch.Tensor: MSE loss value
        """
        return F.mse_loss(q_pred, value_pred)

    def compute_advantages(self, q_values, value_pred):
        """
        Compute advantages and importance weights for policy update
        
        Args:
            q_values: Q-function predictions
            value_pred: Value function predictions
            
        Returns:
            tuple: (advantages, importance_weights)
        """
        advantages = q_values - value_pred
        # Clip advantages to prevent extreme values
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)
        weights = torch.exp(advantages / self.temperature)
        weights = weights / (weights.mean() + 1e-8)  # Normalize weights with epsilon
        return advantages, weights 