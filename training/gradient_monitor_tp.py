"""
Gradient monitoring and adaptive clipping for DeepSeek-V3 with Tensor Parallelism support.
This version is optimized for tensor parallel training, not DDP.
"""
import torch
import torch.nn as nn
from collections import deque, defaultdict
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings

# Check for DTensor support (PyTorch 2.0+)
try:
    from torch.distributed._tensor import DTensor
    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False
    DTensor = None

class GradientMonitorTP:
    """
    Advanced gradient monitoring with adaptive clipping for Tensor Parallel training.
    
    - DTensor-aware gradient computation
    - TP-specific gradient synchronization
    - Optimized for sharded model parameters
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_clip_value: float = 1.0,
        window_size: int = 100,
        warmup_steps: int = 10,
        adaptive_clip: bool = True,
        percentile: float = 95.0,
        clip_factor: float = 1.5,
        logger: Optional[logging.Logger] = None,
        detailed_monitoring: bool = False,
        param_sample_rate: float = 1.0,
        tp_size: int = 1
    ):
        """
        Args:
            model: The model to monitor
            base_clip_value: Base gradient clipping value
            window_size: Window size for moving statistics
            warmup_steps: Steps before enabling adaptive clipping
            adaptive_clip: Whether to use adaptive clipping
            percentile: Percentile for adaptive clipping
            clip_factor: Factor to multiply percentile by for clipping
            logger: Logger for warnings and info
            detailed_monitoring: Enable detailed per-parameter statistics
            param_sample_rate: Fraction of parameters to track detailed stats for
            tp_size: Tensor parallel size (number of GPUs in TP group)
        """
        self.model = model
        self.base_clip_value = base_clip_value
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.adaptive_clip = adaptive_clip
        self.percentile = percentile
        self.clip_factor = clip_factor
        self.logger = logger or logging.getLogger(__name__)
        self.detailed_monitoring = detailed_monitoring
        self.param_sample_rate = param_sample_rate
        self.tp_size = tp_size
        
        # Distributed info for TP
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        
        # Verify TP configuration
        if self.tp_size > 1 and self.world_size % self.tp_size != 0:
            raise ValueError(f"World size ({self.world_size}) must be divisible by TP size ({self.tp_size})")
        
        # Statistics tracking
        self.grad_norm_history = deque(maxlen=window_size)
        self.layer_grad_norms = defaultdict(lambda: deque(maxlen=window_size))
        self.step_count = 0
        
        # Only track detailed stats if enabled
        if self.detailed_monitoring:
            self.param_grad_stats = defaultdict(lambda: deque(maxlen=min(window_size, 10)))
            self._sampled_params = self._sample_parameters()
        
        # Anomaly tracking
        self.nan_count = 0
        self.inf_count = 0
        self.clip_count = 0
        self.extreme_grad_count = 0
        
        # Check for special tensor types
        self._check_tensor_types()
        
        # Layer mapping for better reporting
        self._build_layer_mapping()
    
    def _check_tensor_types(self):
        """Check for FP8 and DTensor parameters"""
        self.has_fp8 = False
        self.has_dtensor = False
        
        for param in self.model.parameters():
            # Check for FP8
            if param.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                self.has_fp8 = True
            
            # Check for DTensor (tensor parallel)
            if HAS_DTENSOR and isinstance(param, DTensor):
                self.has_dtensor = True
        
        if self.has_fp8:
            self.logger.info("FP8 parameters detected, using specialized gradient computation")
        
        if self.has_dtensor:
            self.logger.info("DTensor parameters detected, model is using tensor parallelism")
    
    def _sample_parameters(self) -> set:
        """Sample a subset of parameters for detailed monitoring"""
        all_params = list(self.model.named_parameters())
        num_to_sample = max(1, int(len(all_params) * self.param_sample_rate))
        
        # Always include the largest and smallest parameters
        param_sizes = [(name, param.numel()) for name, param in all_params]
        param_sizes.sort(key=lambda x: x[1])
        
        sampled = set()
        # Add largest and smallest
        if len(param_sizes) > 0:
            sampled.add(param_sizes[0][0])  # smallest
            sampled.add(param_sizes[-1][0])  # largest
        
        # For TP, also sample from each layer type
        layer_params = defaultdict(list)
        for name, size in param_sizes:
            layer_type = self._get_layer_type(name)
            layer_params[layer_type].append(name)
        
        # Sample from each layer type
        import random
        for layer_type, params in layer_params.items():
            if params and len(sampled) < num_to_sample:
                sampled.add(random.choice(params))
        
        self.logger.info(f"Monitoring detailed stats for {len(sampled)}/{len(all_params)} parameters")
        return sampled
    
    def _get_layer_type(self, name: str) -> str:
        """Get layer type from parameter name"""
        if 'embed' in name or 'tok_embeddings' in name:
            return 'embedding'
        elif 'attn' in name or 'attention' in name:
            return 'attention'
        elif 'ffn' in name or 'mlp' in name or 'feed_forward' in name:
            return 'mlp'
        elif 'gate' in name:
            return 'moe_gate'
        elif 'expert' in name:
            return 'moe_expert'
        elif 'norm' in name:
            return 'normalization'
        elif 'head' in name or 'output' in name:
            return 'output_head'
        elif 'mtp' in name:
            return 'mtp_head'
        else:
            return 'other'
    
    def _build_layer_mapping(self):
        """Build a mapping of parameter names to layer types"""
        self.layer_types = {}
        for name, _ in self.model.named_parameters():
            self.layer_types[name] = self._get_layer_type(name)
    
    def _compute_grad_norm_tensor_parallel(self, param: torch.Tensor) -> float:
        """Compute gradient norm for tensor parallel parameters"""
        if param.grad is None:
            return 0.0
        
        grad = param.grad
        
        # Handle DTensor gradients
        if HAS_DTENSOR and isinstance(param, DTensor):
            # DTensor gradients are already properly handled
            # Just compute local norm
            if grad.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                param_norm = grad.float().norm(2)
            else:
                param_norm = grad.norm(2)
        else:
            # Regular tensor
            if grad.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                param_norm = grad.float().norm(2)
            else:
                param_norm = grad.norm(2)
        
        return param_norm.item()
    
    def compute_grad_stats(self, detailed: bool = None) -> Dict[str, float]:
        """Compute gradient statistics for tensor parallel model"""
        if detailed is None:
            detailed = self.detailed_monitoring
            
        stats = {
            'total_norm': 0.0,
            'num_params': 0,
            'num_zero_grad': 0,
            'num_nan': 0,
            'num_inf': 0,
            'tp_size': self.tp_size,
            'has_dtensor': self.has_dtensor
        }
        
        # Per-layer statistics
        layer_norms = defaultdict(list)
        param_norms = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                
                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    stats['num_nan'] += 1
                    self.nan_count += 1
                    if self.rank == 0:  # Only log on rank 0
                        self.logger.warning(f"NaN gradient detected in {name}")
                    continue
                
                if torch.isinf(grad).any():
                    stats['num_inf'] += 1
                    self.inf_count += 1
                    if self.rank == 0:
                        self.logger.warning(f"Inf gradient detected in {name}")
                    continue
                
                # Compute norm with TP awareness
                param_norm = self._compute_grad_norm_tensor_parallel(param)
                param_norms.append(param_norm)
                
                # Track by layer type
                layer_type = self.layer_types.get(name, 'other')
                layer_norms[layer_type].append(param_norm)
                
                # Update stats
                stats['num_params'] += 1
                if param_norm == 0:
                    stats['num_zero_grad'] += 1
                
                # Store detailed parameter stats if enabled and sampled
                if detailed and name in self._sampled_params:
                    self.param_grad_stats[name].append({
                        'norm': param_norm,
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'is_dtensor': isinstance(param, DTensor) if HAS_DTENSOR else False
                    })
        
        # Compute aggregate statistics
        if param_norms:
            # For TP, we need to consider that gradients might be distributed
            local_norm_squared = sum(p**2 for p in param_norms)
            
            if self.tp_size > 1 and torch.distributed.is_initialized():
                # All-reduce the squared norms across TP ranks
                total_norm_squared = torch.tensor(local_norm_squared, device='cuda')
                torch.distributed.all_reduce(total_norm_squared, op=torch.distributed.ReduceOp.SUM)
                stats['total_norm'] = np.sqrt(total_norm_squared.item())
            else:
                stats['total_norm'] = np.sqrt(local_norm_squared)
            
            stats['mean_norm'] = np.mean(param_norms)
            stats['max_norm'] = max(param_norms)
            stats['min_norm'] = min(param_norms)
        
        # Layer-wise statistics
        stats['layer_stats'] = {}
        for layer_type, norms in layer_norms.items():
            if norms:
                self.layer_grad_norms[layer_type].append(np.mean(norms))
                stats['layer_stats'][layer_type] = {
                    'mean': np.mean(norms),
                    'max': max(norms),
                    'count': len(norms)
                }
        
        return stats
    
    def get_adaptive_clip_value(self) -> float:
        """Calculate adaptive clipping value based on gradient history"""
        if not self.adaptive_clip or self.step_count < self.warmup_steps:
            return self.base_clip_value
        
        if len(self.grad_norm_history) < 10:
            return self.base_clip_value
        
        # Use percentile of recent gradient norms
        recent_norms = list(self.grad_norm_history)
        clip_value = np.percentile(recent_norms, self.percentile) * self.clip_factor
        
        # Bound the adaptive value
        clip_value = max(self.base_clip_value * 0.1, clip_value)  # At least 10% of base
        clip_value = min(self.base_clip_value * 10, clip_value)   # At most 10x base
        
        return clip_value
    
    def clip_and_monitor(
        self, 
        max_norm: Optional[float] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        detailed_stats: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Clip gradients and return monitoring statistics for tensor parallel model.
        
        Args:
            max_norm: Maximum norm for clipping (uses adaptive if None)
            scaler: GradScaler for mixed precision training
            detailed_stats: Override detailed monitoring for this step
            optimizer: Optimizer (required if using scaler)
            
        Returns:
            Tuple of (grad_norm, stats_dict)
        """
        self.step_count += 1
        
        # Handle mixed precision - unscale if needed
        if scaler is not None and scaler.is_enabled():
            if optimizer is None:
                raise ValueError("Optimizer must be provided when using GradScaler")
            scaler.unscale_(optimizer)
        
        # Compute gradient statistics
        stats = self.compute_grad_stats(detailed=detailed_stats)
        pre_clip_norm = stats['total_norm']
        
        # Store norm history
        self.grad_norm_history.append(pre_clip_norm)
        
        # Determine clip value
        if max_norm is None:
            max_norm = self.get_adaptive_clip_value()
        
        # Clip gradients
        # For tensor parallel models, clip_grad_norm_ handles DTensors correctly
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm,
            error_if_nonfinite=False
        )
        
        # Track clipping events
        if pre_clip_norm > max_norm:
            self.clip_count += 1
            clip_ratio = pre_clip_norm / max_norm
            stats['clipped'] = True
            stats['clip_ratio'] = clip_ratio
            
            if clip_ratio > 10:
                self.extreme_grad_count += 1
                if self.rank == 0:
                    self.logger.warning(
                        f"Extreme gradient clipping at step {self.step_count}: "
                        f"pre_clip={pre_clip_norm:.2f}, post_clip={grad_norm:.2f}, "
                        f"ratio={clip_ratio:.2f}"
                    )
        else:
            stats['clipped'] = False
            stats['clip_ratio'] = 1.0
        
        # Add monitoring stats
        stats['grad_norm'] = grad_norm
        stats['clip_value'] = max_norm
        stats['clip_rate'] = self.clip_count / self.step_count
        stats['nan_rate'] = self.nan_count / self.step_count
        stats['inf_rate'] = self.inf_count / self.step_count
        
        # Add moving averages
        if len(self.grad_norm_history) > 0:
            stats['grad_norm_ma'] = np.mean(list(self.grad_norm_history))
            stats['grad_norm_std'] = np.std(list(self.grad_norm_history))
        
        # Check for training instability (only on rank 0)
        if self.rank == 0:
            self._check_stability(stats)
        
        return grad_norm, stats
    
    def _check_stability(self, stats: Dict[str, float]):
        """Check for signs of training instability"""
        # High clipping rate
        if stats['clip_rate'] > 0.5 and self.step_count > 100:
            self.logger.warning(
                f"High gradient clipping rate: {stats['clip_rate']:.2%} "
                f"Consider reducing learning rate or increasing clip value"
            )
        
        # Rising gradient norms
        if len(self.grad_norm_history) >= self.window_size:
            recent_mean = np.mean(list(self.grad_norm_history)[-20:])
            older_mean = np.mean(list(self.grad_norm_history)[-100:-80])
            if recent_mean > older_mean * 2:
                self.logger.warning(
                    f"Gradient norms increasing: recent={recent_mean:.2f}, "
                    f"older={older_mean:.2f}"
                )
        
        # Layer-specific issues
        for layer_type, norms in self.layer_grad_norms.items():
            if len(norms) >= 10:
                recent_layer_mean = np.mean(list(norms)[-10:])
                if recent_layer_mean < 1e-7:
                    self.logger.warning(f"Vanishing gradients in {layer_type} layers")
                elif recent_layer_mean > 100:
                    self.logger.warning(f"Exploding gradients in {layer_type} layers")
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of gradient statistics"""
        summary = {
            'total_steps': self.step_count,
            'clip_count': self.clip_count,
            'clip_rate': self.clip_count / max(1, self.step_count),
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'extreme_grad_count': self.extreme_grad_count,
            'tp_size': self.tp_size,
            'has_dtensor': self.has_dtensor,
        }
        
        if self.grad_norm_history:
            summary['grad_norm_stats'] = {
                'mean': np.mean(list(self.grad_norm_history)),
                'std': np.std(list(self.grad_norm_history)),
                'max': max(self.grad_norm_history),
                'min': min(self.grad_norm_history),
                'recent_mean': np.mean(list(self.grad_norm_history)[-10:])
            }
        
        # Layer statistics
        summary['layer_grad_norms'] = {}
        for layer_type, norms in self.layer_grad_norms.items():
            if norms:
                summary['layer_grad_norms'][layer_type] = {
                    'mean': np.mean(list(norms)),
                    'recent': list(norms)[-1] if norms else 0
                }
        
        return summary
    
    def should_stop_training(self) -> bool:
        """Check if training should be stopped due to instability"""
        # Too many NaN/Inf
        if self.nan_count > 10 or self.inf_count > 10:
            return True
        
        # Consistent extreme clipping
        if self.step_count > 100 and self.extreme_grad_count / self.step_count > 0.1:
            return True
        
        # Gradient explosion
        if self.grad_norm_history and max(self.grad_norm_history) > 1000:
            return True
        
        return False
    
    def reset_statistics(self):
        """Reset all statistics (useful for training phase transitions)"""
        self.grad_norm_history.clear()
        self.layer_grad_norms.clear()
        if self.detailed_monitoring:
            self.param_grad_stats.clear()
        self.nan_count = 0
        self.inf_count = 0
        self.clip_count = 0
        self.extreme_grad_count = 0
        self.step_count = 0
