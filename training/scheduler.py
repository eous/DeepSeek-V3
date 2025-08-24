"""
Learning rate and training schedulers for DeepSeek-V3.

This module provides:
- DeepSeekV3LRScheduler: Learning rate scheduler with warmup and cosine decay
- DeepSeekV3TrainingScheduler: Comprehensive scheduler that coordinates LR, MTP, and bias schedules
- Utility functions for common learning rate schedules
"""

import math
from typing import Optional, Dict, Any
import torch
from torch.optim import Optimizer

try:
    from .load_balance_manager import LoadBalanceManager
except ImportError:
    # Handle direct script execution
    from load_balance_manager import LoadBalanceManager


class DeepSeekV3LRScheduler:
    """
    Learning rate scheduler following DeepSeek-V3's multi-stage training schedule.
    
    From the paper:
    - Linear warmup to peak learning rate
    - Stable phase at peak learning rate
    - Cosine decay to final learning rate
    - Coordinated with other hyperparameter schedules (MTP lambda, bias update speed)
    """
    
    def __init__(self, 
                 optimizer: Optimizer,
                 warmup_steps: int = 2000,
                 stable_steps: Optional[int] = None,
                 decay_steps: Optional[int] = None,
                 peak_lr: float = 4.2e-4,
                 final_lr: float = 4.2e-5,
                 total_steps: Optional[int] = None,
                 min_lr_ratio: float = 0.01,
                 initial_lr: Optional[float] = None):
        """
        Initialize the learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer to schedule
            warmup_steps: Number of warmup steps (default: 2000)
            stable_steps: Number of steps at peak LR (default: 20% of total)
            decay_steps: Number of decay steps (default: remaining steps after warmup+stable)
            peak_lr: Peak learning rate after warmup (default: 4.2e-4 from paper)
            final_lr: Final learning rate after decay (default: 4.2e-5, 10% of peak)
            total_steps: Total training steps (used to auto-calculate stable/decay if not provided)
            min_lr_ratio: Minimum LR ratio during warmup to ensure learning starts
            initial_lr: Override the initial LR (default: peak_lr * min_lr_ratio)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.initial_lr = initial_lr if initial_lr is not None else peak_lr * min_lr_ratio
        self.current_step = 0
        
        # Calculate stable and decay steps if not provided
        if total_steps is not None:
            if stable_steps is None:
                # Default: 20% of total steps for stable phase
                stable_steps = int(0.2 * total_steps)
            if decay_steps is None:
                # Remaining steps after warmup and stable
                decay_steps = total_steps - warmup_steps - stable_steps
        
        self.stable_steps = stable_steps or 0
        self.decay_steps = decay_steps
        
        # Initialize optimizer LR
        self._update_lr(self.get_lr(0))
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """
        Calculate learning rate for a given step.
        
        Implements:
        - Linear warmup with minimum floor
        - Stable phase at peak LR
        - Cosine decay to final_lr
        """
        if step is None:
            step = self.current_step
        
        # Use the standalone function for consistency
        return get_warmup_stable_decay_lr(
            step=step,
            warmup_steps=self.warmup_steps,
            stable_steps=self.stable_steps,
            decay_steps=self.decay_steps,
            peak_lr=self.peak_lr,
            final_lr=self.final_lr,
            min_lr_ratio=self.min_lr_ratio
        )
    
    def _update_lr(self, lr: float):
        """Update the learning rate in all parameter groups."""
        for param_group in self.optimizer.param_groups:
            # Handle 8-bit optimizer that requires tensor lr
            if isinstance(param_group['lr'], torch.Tensor):
                param_group['lr'].fill_(lr)
            else:
                param_group['lr'] = lr
    
    def step(self):
        """Update learning rate for the current step."""
        self.current_step += 1
        lr = self.get_lr()
        self._update_lr(lr)
        return lr
    
    def get_last_lr(self) -> float:
        """Get the last computed learning rate."""
        return self.get_lr()
    
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'peak_lr': self.peak_lr,
            'final_lr': self.final_lr,
            'total_steps': self.total_steps,
            'min_lr_ratio': self.min_lr_ratio
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.peak_lr = state_dict['peak_lr']
        self.final_lr = state_dict['final_lr']
        self.total_steps = state_dict.get('total_steps')
        self.min_lr_ratio = state_dict.get('min_lr_ratio', 0.1)
        
        # Update current LR
        self._update_lr(self.get_lr())


class DeepSeekV3TrainingScheduler:
    """
    Comprehensive training scheduler for DeepSeek-V3 that coordinates:
    - Learning rate schedule
    - MTP lambda schedule (0.3 → 0.1 after 70%)
    - Bias update speed schedule (0.001 → 0.0 after 90%)
    """
    
    def __init__(self,
                 model,
                 optimizer: Optimizer,
                 load_balance_manager: LoadBalanceManager,
                 warmup_steps: int = 2000,
                 stable_steps: Optional[int] = None,
                 decay_steps: Optional[int] = None,
                 peak_lr: float = 4.2e-4,
                 final_lr: float = 4.2e-5,
                 total_steps: Optional[int] = None,
                 mtp_transition_step: Optional[int] = None,
                 bias_freeze_step: Optional[int] = None):
        """
        Initialize comprehensive training scheduler.
        
        Args:
            model: The Transformer model
            optimizer: PyTorch optimizer
            load_balance_manager: Load balance manager for bias updates
            warmup_steps: LR warmup steps
            stable_steps: Number of steps at peak LR (default: 20% of total)
            decay_steps: Number of decay steps (default: remaining after warmup+stable)
            peak_lr: Peak learning rate
            final_lr: Final learning rate
            total_steps: Total training steps (used to auto-calculate stable/decay if not provided)
            mtp_transition_step: Step to transition MTP lambda from 0.3 to 0.1
            bias_freeze_step: Step to freeze bias updates (γ = 0.0)
        """
        self.model = model
        self.optimizer = optimizer
        self.load_balance_manager = load_balance_manager
        
        # LR scheduler with warmup + stable + decay
        self.lr_scheduler = DeepSeekV3LRScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            peak_lr=peak_lr,
            final_lr=final_lr,
            total_steps=total_steps
        )
        
        # Schedule milestones
        self.mtp_transition_step = mtp_transition_step
        self.bias_freeze_step = bias_freeze_step
        
        # Initial values
        self.initial_mtp_lambda = 0.3
        self.final_mtp_lambda = 0.1
        self.initial_bias_speed = 0.001
        
    def step(self) -> Dict[str, float]:
        """
        Update all scheduled parameters for the current training step.
        
        Returns:
            Dictionary with current values of all scheduled parameters
        """
        current_step = self.lr_scheduler.current_step
        
        # Update learning rate
        lr = self.lr_scheduler.step()
        
        # Update MTP lambda if applicable
        mtp_lambda = self.initial_mtp_lambda
        if hasattr(self.model, 'mtp_lambda') and self.mtp_transition_step is not None:
            if current_step >= self.mtp_transition_step:
                mtp_lambda = self.final_mtp_lambda
            self.model.mtp_lambda = mtp_lambda
        
        # Update bias update speed
        bias_speed = self.initial_bias_speed
        if self.bias_freeze_step is not None and current_step >= self.bias_freeze_step:
            bias_speed = 0.0
            self.load_balance_manager.set_bias_update_speed(bias_speed)
        
        return {
            'lr': lr,
            'mtp_lambda': mtp_lambda,
            'bias_update_speed': bias_speed,
            'step': current_step
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Return full scheduler state for checkpointing."""
        return {
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'mtp_transition_step': self.mtp_transition_step,
            'bias_freeze_step': self.bias_freeze_step,
            'initial_mtp_lambda': self.initial_mtp_lambda,
            'final_mtp_lambda': self.final_mtp_lambda,
            'initial_bias_speed': self.initial_bias_speed
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load full scheduler state from checkpoint."""
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.mtp_transition_step = state_dict.get('mtp_transition_step')
        self.bias_freeze_step = state_dict.get('bias_freeze_step')
        self.initial_mtp_lambda = state_dict.get('initial_mtp_lambda', 0.3)
        self.final_mtp_lambda = state_dict.get('final_mtp_lambda', 0.1)
        self.initial_bias_speed = state_dict.get('initial_bias_speed', 0.001)


# Utility functions for common learning rate schedules

def get_warmup_stable_decay_lr(step: int,
                              warmup_steps: int,
                              stable_steps: int,
                              decay_steps: int,
                              peak_lr: float,
                              final_lr: float,
                              min_lr_ratio: float = 0.1) -> float:
    """
    Get learning rate with warmup, stable, and decay phases.
    
    This schedule has three distinct phases:
    1. Linear warmup to peak_lr
    2. Constant at peak_lr for stable_steps
    3. Cosine decay to final_lr over decay_steps
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        stable_steps: Number of steps at peak_lr
        decay_steps: Number of decay steps
        peak_lr: Peak learning rate
        final_lr: Final learning rate
        min_lr_ratio: Minimum LR ratio during warmup
        
    Returns:
        Learning rate for the given step
    """
    if step < warmup_steps:
        # Linear warmup with minimum floor
        warmup_lr = peak_lr * step / warmup_steps
        min_lr = peak_lr * min_lr_ratio
        return max(warmup_lr, min_lr)
    
    if step < warmup_steps + stable_steps:
        # Stable phase
        return peak_lr
    
    # Decay phase
    decay_step = step - warmup_steps - stable_steps
    if decay_step >= decay_steps:
        return final_lr
    
    progress = decay_step / decay_steps
    return final_lr + (peak_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def create_scheduler(scheduler_type: str,
                    optimizer: Optimizer,
                    model=None,
                    load_balance_manager: Optional[LoadBalanceManager] = None,
                    **kwargs) -> DeepSeekV3LRScheduler:
    """
    Factory function to create schedulers.
    
    Args:
        scheduler_type: Type of scheduler ('lr_only' or 'full')
        optimizer: PyTorch optimizer
        model: Model (required for 'full' scheduler)
        load_balance_manager: Load balance manager (required for 'full' scheduler)
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Scheduler instance
    """
    if scheduler_type == 'lr_only':
        return DeepSeekV3LRScheduler(optimizer, **kwargs)
    elif scheduler_type == 'full':
        if model is None or load_balance_manager is None:
            raise ValueError("Model and load_balance_manager required for full scheduler")
        return DeepSeekV3TrainingScheduler(model, optimizer, load_balance_manager, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
