import torch
import numpy as np

def compute_action_accuracy(pred_actions: torch.Tensor, target_actions: torch.Tensor, 
                          thresholds: list[float]) -> dict[str, float]:
    """
    Compute accuracy metrics for continuous action predictions at various thresholds.
    
    Args:
        pred_actions: (B, A) predicted actions
        target_actions: (B, A) target actions
        thresholds: List of absolute error thresholds to check
        
    Returns:
        Dictionary containing:
        - exact_match: % of actions where all dimensions match within smallest threshold
        - per_threshold: % of actions within each threshold
        - per_dim: Mean absolute error for each action dimension
    """
    # Compute absolute errors
    abs_errors = torch.abs(pred_actions - target_actions)
    mean_abs_error = abs_errors.mean().item()
    
    # Per-dimension mean absolute error
    per_dim_mae = abs_errors.mean(dim=0)
    
    metrics = {
        'mean_absolute_error': mean_abs_error,
    }
    
    # Add per-dimension MAE
    for i, mae in enumerate(per_dim_mae):
        metrics[f'mae_dim_{i}'] = mae.item()
    
    # Compute accuracy at different thresholds
    for threshold in thresholds:
        # All dimensions within threshold
        exact_match = (abs_errors <= threshold).all(dim=-1).float().mean().item()
        metrics[f'exact_match/thresh_{threshold:.3f}'] = exact_match
        
        # Average dimensions within threshold
        dim_match = (abs_errors <= threshold).float().mean().item()
        metrics[f'dim_match/thresh_{threshold:.3f}'] = dim_match
    
    return metrics

def compute_success_rate(pred_actions: torch.Tensor, target_actions: torch.Tensor,
                        success_threshold: float = 0.1) -> float:
    """
    Compute the success rate - percentage of actions where all dimensions are within threshold.
    This is a single summary metric for evaluation.
    
    Args:
        pred_actions: (B, A) predicted actions
        target_actions: (B, A) target actions
        success_threshold: Threshold for considering an action successful
        
    Returns:
        Success rate as a float between 0 and 1
    """
    abs_errors = torch.abs(pred_actions - target_actions)
    success = (abs_errors <= success_threshold).all(dim=-1).float().mean().item()
    return success 