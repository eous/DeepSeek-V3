"""
Utility functions for MoE load balancing through expert biases.

This module provides direct access to gate bias adjustments.
Load balancing is achieved by modifying expert selection biases
based on measured load imbalance, without auxiliary loss terms.
"""

import torch
from typing import Iterator
from model import Gate, MoE


__all__ = [
    'get_all_gates',
    'adjust_all_gate_biases', 
    'reset_gate_statistics',
    'get_load_balance_metrics',
    'set_bias_update_speed',
    'get_normalized_loads'
]


def get_all_gates(model: torch.nn.Module) -> Iterator[Gate]:
    """
    Iterate over all Gate modules in a model.
    
    Args:
        model: The model to search for Gate modules
        
    Yields:
        Gate modules found in the model
    """
    for module in model.modules():
        if isinstance(module, Gate):
            yield module


def adjust_all_gate_biases(model: torch.nn.Module) -> None:
    """
    Adjust biases for all gates in the model based on their load statistics.
    
    This should be called at the end of each training step to balance
    expert loads through bias adjustments.
    
    Args:
        model: The model containing Gate modules
    """
    for gate in get_all_gates(model):
        gate.adjust_biases()


def reset_gate_statistics(model: torch.nn.Module) -> None:
    """
    Reset load statistics for all gates in the model.
    
    Args:
        model: The model containing Gate modules
    """
    for gate in get_all_gates(model):
        if hasattr(gate, 'expert_counts'):
            gate.expert_counts.zero_()
        if hasattr(gate, 'expert_loads'):
            gate.expert_loads.zero_()


def get_normalized_loads(loads: torch.Tensor, n_experts: int) -> torch.Tensor:
    """
    Safely normalize loads to get distribution over experts.
    
    Args:
        loads: Expert load tensor
        n_experts: Number of experts
        
    Returns:
        Normalized loads (sums to 1), or uniform distribution if loads are all zero
    """
    total_load = loads.sum()
    if total_load > 0:
        return loads / total_load
    else:
        # Return uniform distribution if no loads yet
        return torch.ones_like(loads) / n_experts


def get_load_balance_metrics(model: torch.nn.Module) -> dict:
    """
    Get load balancing metrics for monitoring.
    
    Args:
        model: The model containing Gate modules
        
    Returns:
        Dictionary with load balancing metrics for each gate
    """
    metrics = {}
    
    for i, gate in enumerate(get_all_gates(model)):
        gate_metrics = {}
        
        if hasattr(gate, 'expert_loads') and gate.expert_loads is not None:
            loads = gate.expert_loads
            gate_metrics['load_mean'] = loads.mean().item()
            gate_metrics['load_std'] = loads.std().item()
            gate_metrics['load_max'] = loads.max().item()
            gate_metrics['load_min'] = loads.min().item()
            gate_metrics['load_variance'] = loads.var().item()
            
            # Add normalized load distribution
            normalized_loads = get_normalized_loads(loads, gate.n_routed_experts)
            gate_metrics['normalized_load_variance'] = normalized_loads.var().item()
            
        if hasattr(gate, 'expert_biases'):
            biases = gate.expert_biases
            gate_metrics['bias_mean'] = biases.mean().item()
            gate_metrics['bias_std'] = biases.std().item()
            gate_metrics['bias_range'] = (biases.max() - biases.min()).item()
            
        metrics[f'gate_{i}'] = gate_metrics
    
    return metrics


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
