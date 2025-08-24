"""
Load Balance Manager for DeepSeek-V3 Auxiliary-Loss-Free Load Balancing

This module provides utilities for managing load balancing across all MoE layers
in a model during training, implementing the auxiliary-loss-free approach from
the DeepSeek-V3 paper.
"""

import torch
from typing import List, Optional
from model import MoE, Gate


class LoadBalanceManager:
    """
    Manages load balancing across all MoE layers in a model.
    
    This class coordinates the bias adjustments and balance loss computation
    for auxiliary-loss-free load balancing as described in DeepSeek-V3.
    
    Attributes:
        model: The transformer model containing MoE layers
        moe_layers: List of MoE modules in the model
        gate_modules: List of Gate modules from each MoE layer
    """
    
    def __init__(self, model):
        """
        Initialize the LoadBalanceManager.
        
        Args:
            model: The transformer model to manage
        """
        self.model = model
        self.moe_layers = []
        self.gate_modules = []
        
        # Find all MoE layers and their gate modules
        for name, module in model.named_modules():
            if isinstance(module, MoE):
                self.moe_layers.append(module)
                if hasattr(module, 'gate') and isinstance(module.gate, Gate):
                    self.gate_modules.append(module.gate)
                    
        print(f"LoadBalanceManager initialized with {len(self.moe_layers)} MoE layers")
    
    def step(self):
        """
        Perform one step of bias adjustment.
        
        This should be called at the end of each training step to update
        the expert biases based on accumulated load statistics.
        """
        for gate in self.gate_modules:
            if hasattr(gate, 'adjust_biases'):
                gate.adjust_biases()
    
    def get_balance_loss(self) -> float:
        """
        Aggregate balance losses from all MoE layers.
        
        Note: This returns a float, not a tensor, to avoid backward graph issues.
        The actual balance loss should be computed during the forward pass.
        
        Returns:
            Total balance loss across all MoE layers as a float
        """
        total_loss = 0.0
        
        for moe in self.moe_layers:
            if hasattr(moe, '_last_balance_loss') and moe._last_balance_loss is not None:
                # Convert to float to detach from computation graph
                if isinstance(moe._last_balance_loss, torch.Tensor):
                    total_loss += moe._last_balance_loss.item()
                else:
                    total_loss += moe._last_balance_loss
                # Clear the stored loss to prevent reuse
                moe._last_balance_loss = None
                
        return total_loss
    
    def reset_statistics(self):
        """Reset all load statistics in gate modules."""
        for gate in self.gate_modules:
            if hasattr(gate, 'expert_counts'):
                gate.expert_counts.zero_()
            if hasattr(gate, 'expert_loads'):
                gate.expert_loads.zero_()
    
    def get_load_balance_stats(self) -> dict:
        """
        Get current load balancing statistics for monitoring.
        
        Returns:
            Dictionary containing load statistics for each MoE layer
        """
        stats = {}
        
        for i, gate in enumerate(self.gate_modules):
            layer_stats = {}
            
            if hasattr(gate, 'expert_loads'):
                layer_stats['expert_loads'] = gate.expert_loads.cpu().numpy().tolist()
                layer_stats['load_variance'] = gate.expert_loads.var().item()
                layer_stats['max_load'] = gate.expert_loads.max().item()
                layer_stats['min_load'] = gate.expert_loads.min().item()
                
            if hasattr(gate, 'expert_biases'):
                layer_stats['bias_mean'] = gate.expert_biases.mean().item()
                layer_stats['bias_std'] = gate.expert_biases.std().item()
                layer_stats['bias_max'] = gate.expert_biases.max().item()
                layer_stats['bias_min'] = gate.expert_biases.min().item()
                
            stats[f'moe_layer_{i}'] = layer_stats
            
        return stats
    
    def set_bias_update_speed(self, speed: float):
        """
        Update the bias adjustment speed for all gate modules.
        
        Args:
            speed: New bias update speed
        """
        for gate in self.gate_modules:
            if hasattr(gate, 'bias_update_speed'):
                gate.bias_update_speed = speed
                
    def enable_training_mode(self):
        """Enable training mode for all MoE and Gate modules."""
        for moe in self.moe_layers:
            moe.train()
        for gate in self.gate_modules:
            gate.train()
            
    def disable_training_mode(self):
        """Disable training mode for all MoE and Gate modules."""
        for moe in self.moe_layers:
            moe.eval()
        for gate in self.gate_modules:
            gate.eval()
