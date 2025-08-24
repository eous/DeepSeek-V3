"""
Utility functions for MoE load balancing through expert biases.

This module provides monitoring utilities for the MoE implementation
where load balancing is achieved through inline bias adjustments.
"""

import torch
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from model import Gate, MoE


__all__ = [
    'get_all_gates',
    'set_bias_update_speed',
    'get_expert_bias_summary'
]


def get_all_gates(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """
    Iterate over all Gate modules in a model.
    
    Args:
        model: The model to search for Gate modules
        
    Yields:
        Gate modules found in the model
    """
    for module in model.modules():
        if module.__class__.__name__ == 'Gate':
            yield module




def set_bias_update_speed(model: torch.nn.Module, speed: float) -> None:
    """
    Update the bias adjustment speed for all gates.
    
    Args:
        model: The model containing Gate modules
        speed: New bias update speed
    """
    for gate in get_all_gates(model):
        if hasattr(gate, 'bias_update_speed'):
            gate.bias_update_speed = speed


def get_expert_bias_summary(model: torch.nn.Module) -> dict:
    """
    Get summary statistics of expert biases across all gates.
    
    Args:
        model: The model containing Gate modules
        
    Returns:
        Dictionary with bias statistics for each gate
    """
    summary = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'expert_biases'):
            biases = module.expert_biases.detach().cpu().float()  # Convert to float32 for numpy
            summary[name] = {
                'mean': biases.mean().item(),
                'std': biases.std().item(),
                'min': biases.min().item(),
                'max': biases.max().item(),
                'range': (biases.max() - biases.min()).item(),
                'biases': biases  # Return as numpy array for compatibility
            }
    
    return summary
