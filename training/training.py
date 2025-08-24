"""
Training Integration Example for DeepSeek-V3 with Tensor Parallelism

This example demonstrates how to integrate DeepSeek-V3 with tensor parallelism (TP)
for multi-GPU training, which is more suitable than DDP for large models.
"""

import os
import sys

# Suppress output for non-rank-0 processes early to avoid interfering with rich UI
_is_rank_zero = os.environ.get("LOCAL_RANK", "0") == "0"
if not _is_rank_zero:
    # Redirect stdout to devnull for non-rank-0 processes
    _original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
else:
    print("Starting...")

# IMPORTANT: Set environment variables BEFORE importing PyTorch

# CUDA Memory Allocation Optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable asynchronous operations
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Enable async error handling

# PyTorch Memory Allocator Configuration
alloc_conf_settings = [
    'expandable_segments:True',
    'garbage_collection_threshold:0.8',
    'max_split_size_mb:256',
    'roundup_power2_divisions:16',
]
os.environ['TORCH_ALLOC_CONF'] = ','.join(alloc_conf_settings)

# Additional CUDA optimizations
os.environ['PYTORCH_NVFUSER_DISABLE'] = '0'
os.environ['CUBLAS_EMULATION_STRATEGY'] = 'performant'


# CPU thread optimization for multi-process training
# Reduce threads per process to avoid oversubscription
# Removed GOMP_CPU_AFFINITY - invalid value 'N-M' causes import torch to hang
# os.environ['GOMP_CPU_AFFINITY'] = 'N-M'
os.environ["OMP_NUM_THREADS"] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

if _is_rank_zero:
    print("Set environment variables for CPU and CUDA optimizations")

# Now import PyTorch and other libraries
import torch
if _is_rank_zero:
    print("PyTorch version:", torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


from torchao.optim import AdamW8bit
from torchao.optim import CPUOffloadOptimizer

if _is_rank_zero:
    print("Torchao optimizers imported")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Optional, Tuple, List
import json
import math
import logging
import argparse
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from collections import defaultdict

if _is_rank_zero:
    print("Loaded core libraries")

# Import rich for better UI
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
    # Force color output and terminal detection for torchrun environments
    console = Console(force_terminal=True, force_interactive=True, color_system="truecolor")
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Only print on rank 0 or when not in distributed mode
if not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
    print("Rich UI libraries imported" if RICH_AVAILABLE else "Rich UI libraries not available, using plain text output")

# Import model components
from model import Transformer, ModelArgs, compute_mtp_loss, set_tensor_parallel_config
from scheduler import DeepSeekV3TrainingScheduler, DeepSeekV3LRScheduler
import moe_utils
from dataset import ParquetTextDataset
from gradient_monitor_tp import GradientMonitorTP

if _is_rank_zero:
    print("Model components imported")

# Import tensor parallel utilities
try:
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
        PrepareModuleInput,
        PrepareModuleOutput,
    )
    from torch.distributed._tensor import DTensor, Replicate, Shard
    from torch.distributed.device_mesh import init_device_mesh
    HAS_TP = True
except ImportError:
    HAS_TP = False
    print("Warning: Tensor parallelism not available. Install PyTorch 2.0+ with distributed support.")

print("PyTorch version:", torch.__version__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if _is_rank_zero:
    print("PyTorch backend configured for TF32")

@dataclass
class TPConfig:
    """Tensor Parallelism Configuration"""
    tp_size: int = 1  # Number of GPUs for tensor parallelism
    pp_size: int = 1  # Number of GPUs for pipeline parallelism (future)
    sequence_parallel: bool = True  # Enable sequence parallelism
    tensor_parallel_mode: str = "column"  # column or row split

def setup_distributed(args):
    """Setup distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    return rank, world_size, local_rank

def create_device_mesh(world_size: int, tp_size: int):
    """Create device mesh for tensor parallelism"""
    if not HAS_TP:
        return None
    
    # For now, we use all GPUs for tensor parallelism
    # Future: support hybrid TP + PP
    device_mesh = init_device_mesh("cuda", (tp_size,))
    return device_mesh

class TensorParallelDeepSeekV3Model(Transformer):
    """DeepSeekV3 Model with Tensor Parallelism support"""
    
    def __init__(self, args: ModelArgs, device_mesh=None):
        # Set tensor parallel configuration before calling parent init
        if device_mesh is not None:
            tp_size = device_mesh.size()
            tp_rank = dist.get_rank() if dist.is_initialized() else 0
            set_tensor_parallel_config(tp_size, tp_rank)
        
        super().__init__(args)
        self.device_mesh = device_mesh
        self.tp_size = device_mesh.size() if device_mesh else 1
        self._is_tensor_parallel = True  # Flag to indicate TP mode
        
    def parallelize_layers(self):
        """Apply tensor parallelism to model layers"""
        if not HAS_TP or self.device_mesh is None:
            return
        
        # Parallelize embedding layer
        parallelize_module(
            self.tok_embeddings,
            self.device_mesh,
            ColwiseParallel(output_layouts=Shard(1)),
        )
        
        # Parallelize each transformer layer
        for layer_idx, layer in enumerate(self.layers):
            # Parallelize attention
            if hasattr(layer, 'attn'):
                # Q, K, V projections - column-wise
                parallelize_module(
                    layer.attn.q_proj,
                    self.device_mesh,
                    ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                )
                parallelize_module(
                    layer.attn.k_proj,
                    self.device_mesh,
                    ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                )
                parallelize_module(
                    layer.attn.v_proj,
                    self.device_mesh,
                    ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                )
                
                # Output projection - row-wise
                parallelize_module(
                    layer.attn.o_proj,
                    self.device_mesh,
                    RowwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
                )
            
            # Parallelize MoE layers
            if hasattr(layer, 'moe') and layer.moe is not None:
                # Expert parallelism is more complex
                # For now, we shard experts across GPUs
                num_experts_per_gpu = layer.moe.num_experts // self.tp_size
                
                # Each GPU handles a subset of experts
                # This requires custom logic in the forward pass
                layer.moe.num_experts_per_gpu = num_experts_per_gpu
                layer.moe.tp_rank = dist.get_rank() if dist.is_initialized() else 0
                layer.moe.tp_size = self.tp_size
        
        # Parallelize output layer
        parallelize_module(
            self.output,
            self.device_mesh,
            ColwiseParallel(input_layouts=Replicate()),
        )
        
        # Parallelize MTP heads if present
        if hasattr(self, 'mtp_heads') and self.mtp_heads is not None:
            for head in self.mtp_heads:
                parallelize_module(
                    head,
                    self.device_mesh,
                    ColwiseParallel(input_layouts=Replicate()),
                )

def create_tp_model(config: Dict, tp_config: TPConfig, device_mesh):
    """Create model with tensor parallelism"""
    # Convert config dict to ModelArgs
    model_args = ModelArgs(
        vocab_size=config.get('vocab_size', 128815),
        dim=config['dim'],
        inter_dim=config.get('inter_dim', int(config['dim'] * 2.75)),
        moe_inter_dim=config.get('moe_inter_dim', int(config['dim'] * 0.6875)),
        n_layers=config['n_layers'],
        n_dense_layers=config.get('n_dense_layers', 1),
        n_heads=config['n_heads'],
        n_routed_experts=config.get('n_routed_experts', 8),
        n_shared_experts=config.get('n_shared_experts', 2),
        n_activated_experts=config.get('n_activated_experts', 2),
        n_expert_groups=config.get('n_expert_groups', 1),
        n_limited_groups=config.get('n_limited_groups', 1),
        score_func=config.get('score_func', 'softmax'),
        route_scale=config.get('route_scale', 1.0),
        q_lora_rank=config.get('q_lora_rank', int(config['dim'] * 0.75)),
        kv_lora_rank=config.get('kv_lora_rank', int(config['dim'] * 0.25)),
        qk_nope_head_dim=config.get('qk_nope_head_dim', 128),
        qk_rope_head_dim=config.get('qk_rope_head_dim', 64),
        v_head_dim=config.get('v_head_dim', 128),
        mscale=config.get('mscale', 0.707),
        bias_update_speed=config.get('bias_update_speed', 0.001),
        initializer_range=config.get('initializer_range', 0.02),
        mtp_depth=config.get('mtp_depth', 1),
        mtp_lambda=config.get('mtp_lambda', 0.3),
        max_seq_len=config.get('max_seq_len', 4096),
        max_batch_size=config.get('max_batch_size', 8),
        gradient_checkpointing=config.get('gradient_checkpointing', True),
        gradient_checkpointing_use_reentrant=config.get('gradient_checkpointing_use_reentrant', False)
    )
    
    # Note: For TP, we don't scale dimensions here as the parallelization
    # handles the distribution of parameters across devices
    
    # Only use TensorParallelDeepSeekV3Model when actually using tensor parallelism
    if device_mesh is not None:
        model = TensorParallelDeepSeekV3Model(model_args, device_mesh)
        model.parallelize_layers()
    else:
        # For single GPU (TP=1), use regular Transformer model
        model = Transformer(model_args)
    
    return model

def all_reduce_gradients(model: nn.Module, world_size: int):
    """All-reduce gradients across tensor parallel ranks"""
    if world_size == 1:
        return
    
    # All-reduce gradients that need synchronization
    for param in model.parameters():
        if param.grad is not None:
            # For DTensor, gradients are already synchronized
            if not isinstance(param, DTensor):
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)

def estimate_params(config: dict) -> int:
    """Accurately estimate total model parameters for DeepSeek-V3"""
    dim = config.get('dim', 2048)
    n_layers = config.get('n_layers', 12)
    n_heads = config.get('n_heads', 16)
    vocab_size = config.get('vocab_size', 128256)
    
    # MLA (Multi-head Latent Attention) parameters
    q_lora_rank = config.get('q_lora_rank', int(dim * 0.75))
    kv_lora_rank = config.get('kv_lora_rank', int(dim * 0.25))
    qk_nope_head_dim = config.get('qk_nope_head_dim', 128)
    qk_rope_head_dim = config.get('qk_rope_head_dim', 64)
    v_head_dim = config.get('v_head_dim', 128)
    
    # MoE parameters
    n_routed_experts = config.get('n_routed_experts', 16)
    n_shared_experts = config.get('n_shared_experts', 2)
    n_dense_layers = config.get('n_dense_layers', 1)
    inter_dim = config.get('inter_dim', int(dim * 2.75))
    moe_inter_dim = config.get('moe_inter_dim', int(dim * 0.6875))
    
    # MTP parameters
    mtp_depth = config.get('mtp_depth', 1)
    
    total_params = 0
    
    # 1. Embedding layer (also used as output projection via weight tying)
    embed_params = vocab_size * dim
    total_params += embed_params
    
    # 2. Layer norms (RMSNorm has no bias)
    # Each layer has 2 norms (pre and post), plus final norm
    norm_params = dim * (2 * n_layers + 1)
    total_params += norm_params
    
    # 3. Attention parameters per layer (MLA architecture)
    for layer_idx in range(n_layers):
        # Compressed KV with LoRA
        # wkv_a: dim -> (kv_lora_rank + qk_rope_head_dim)
        kv_a_params = dim * (kv_lora_rank + qk_rope_head_dim)
        # wkv_b: kv_lora_rank -> n_heads * (qk_nope_head_dim + v_head_dim)
        kv_b_params = kv_lora_rank * (n_heads * (qk_nope_head_dim + v_head_dim))
        
        # Query projection with LoRA  
        # wq_a: dim -> q_lora_rank
        q_a_params = dim * q_lora_rank
        # wq_b: q_lora_rank -> n_heads * (qk_nope_head_dim + qk_rope_head_dim)
        q_b_params = q_lora_rank * (n_heads * (qk_nope_head_dim + qk_rope_head_dim))
        
        # Output projection
        o_proj_params = (n_heads * v_head_dim) * dim
        
        # Normalization layers in attention
        norm_params = q_lora_rank + kv_lora_rank  # q_norm + kv_norm weights
        
        attn_params = kv_a_params + kv_b_params + q_a_params + q_b_params + o_proj_params + norm_params
        total_params += attn_params
    
    # 4. FFN/MoE parameters per layer
    for layer_idx in range(n_layers):
        if layer_idx < n_dense_layers:
            # Dense FFN layer
            ffn_params = dim * inter_dim * 3  # gate, up, down projections
            total_params += ffn_params
        else:
            # MoE layer
            # Shared experts (use moe_inter_dim, not inter_dim)
            if n_shared_experts > 0:
                shared_params = dim * (n_shared_experts * moe_inter_dim) * 3
                total_params += shared_params
            
            # Routed experts  
            routed_params = n_routed_experts * dim * moe_inter_dim * 3
            total_params += routed_params
            
            # Gating network (linear layer + expert biases)
            gate_params = dim * n_routed_experts + n_routed_experts  # weight + expert_biases
            total_params += gate_params
    
    # 5. MTP heads if enabled
    if mtp_depth > 0:
        # Each MTP module has:
        # - A projection layer (dim -> vocab_size, but shares weights with main output)
        # - A transformer block (attention + FFN)
        # - Two norm layers (norm_h, norm_emb)
        
        for _ in range(mtp_depth):
            # MTP transformer block (similar to a regular transformer layer)
            # Attention params (same as regular layer)
            mtp_attn_params = (
                dim * (kv_lora_rank + qk_rope_head_dim) +  # wkv_a
                kv_lora_rank * (n_heads * (qk_nope_head_dim + v_head_dim)) +  # wkv_b
                dim * q_lora_rank +  # wq_a
                q_lora_rank * (n_heads * (qk_nope_head_dim + qk_rope_head_dim)) +  # wq_b
                (n_heads * v_head_dim) * dim +  # o_proj
                q_lora_rank + kv_lora_rank  # q_norm + kv_norm
            )
            
            # FFN params (dense layer, not MoE)
            mtp_ffn_params = dim * inter_dim * 3  # w1, w2, w3
            
            # Norm layers
            mtp_norm_params = dim * 4  # attn_norm, ffn_norm, norm_h, norm_emb
            
            # Projection layer (2*dim -> dim) for combining h_prev and next_emb
            mtp_proj_params = (2 * dim) * dim
            
            # Note: Output projection to vocab uses the main model's head (weight tied)
            
            total_params += mtp_attn_params + mtp_ffn_params + mtp_norm_params + mtp_proj_params
    
    return int(total_params)


class CheckpointManager:
    """Modern checkpoint management with atomic saves and automatic cleanup."""
    
    def __init__(self, save_dir: str, keep_checkpoints: int = 3, rank: int = 0):
        self.save_dir = save_dir
        self.keep_checkpoints = keep_checkpoints
        self.rank = rank
        self.checkpoint_list = []
        
        if self.rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            # Load existing checkpoint list
            self._refresh_checkpoint_list()
    
    def _refresh_checkpoint_list(self):
        """Refresh the list of existing checkpoints."""
        if os.path.exists(self.save_dir):
            checkpoints = [f for f in os.listdir(self.save_dir) 
                          if f.startswith('checkpoint_step_') and f.endswith('.pt')]
            # Sort by step number
            self.checkpoint_list = sorted(checkpoints, 
                                        key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    def save_checkpoint(self, state_dict: dict, global_step: int, metrics: dict = None):
        """Save checkpoint with atomic write and manage checkpoint history."""
        if self.rank != 0:
            return
        
        checkpoint_name = f'checkpoint_step_{global_step}.pt'
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        temp_path = checkpoint_path + '.tmp'
        
        # Add metadata
        state_dict['metadata'] = {
            'timestamp': time.time(),
            'step': global_step,
            'metrics': metrics or {}
        }
        
        # Save to temporary file first (atomic write)
        torch.save(state_dict, temp_path)
        # Rename to final path (atomic on most filesystems)
        os.rename(temp_path, checkpoint_path)
        
        # Update checkpoint list
        self.checkpoint_list.append(checkpoint_name)
        
        # Save latest symlink
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(checkpoint_name, latest_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        logging.info(f"Saved checkpoint: {checkpoint_path}")
        if metrics:
            logging.info(f"Checkpoint metrics: {metrics}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        if len(self.checkpoint_list) > self.keep_checkpoints:
            checkpoints_to_remove = self.checkpoint_list[:-self.keep_checkpoints]
            for checkpoint in checkpoints_to_remove:
                path = os.path.join(self.save_dir, checkpoint)
                if os.path.exists(path):
                    os.remove(path)
                    logging.info(f"Removed old checkpoint: {path}")
            self.checkpoint_list = self.checkpoint_list[-self.keep_checkpoints:]
    
    def get_latest_checkpoint(self):
        """Get the path to the latest checkpoint."""
        if not self.checkpoint_list:
            return None
        latest = self.checkpoint_list[-1]
        return os.path.join(self.save_dir, latest)
    
    def load_checkpoint(self, checkpoint_path: str = None):
        """Load a checkpoint, either specified or latest."""
        if checkpoint_path == 'latest' or checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                return None
        
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint


def resume_training(args, model, optimizer, scheduler=None):
    """Resume training from a checkpoint."""
    checkpoint_manager = CheckpointManager(args.save_dir, args.keep_checkpoints)
    
    checkpoint_path = args.resume
    if checkpoint_path:
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
        if checkpoint is None:
            logging.error("Failed to load checkpoint, starting from scratch")
            return 0, 0
        
        # Load model state (handle DDP wrapper)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Loaded model state")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("Loaded optimizer state")
        
        # Load scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("Loaded scheduler state")
        
        # Get training progress
        global_step = checkpoint.get('global_step', 0)
        start_epoch = checkpoint.get('epoch', 0)
        
        # Log resume info
        logging.info(f"Resuming from step {global_step}, epoch {start_epoch}")
        if 'metadata' in checkpoint:
            meta = checkpoint['metadata']
            logging.info(f"Checkpoint timestamp: {time.ctime(meta['timestamp'])}")
            if 'metrics' in meta:
                logging.info(f"Checkpoint metrics: {meta['metrics']}")
        
        return global_step, start_epoch
    
    return 0, 0


def main():
    parser = argparse.ArgumentParser(description='DeepSeek-V3 Training with Tensor Parallelism')
    
    # Model configuration arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model config JSON file (overrides other model args)')
    parser.add_argument('--dim', type=int, default=2048, 
                       help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=16, 
                       help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--n-routed-experts', type=int, default=16,
                       help='Number of routed experts for MoE')
    parser.add_argument('--n-activated-experts', type=int, default=2,
                       help='Number of activated experts')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps')
    parser.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum training steps')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=1000,
                       help='Checkpoint save interval')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (or "latest" for most recent)')
    parser.add_argument('--keep-checkpoints', type=int, default=3,
                       help='Number of recent checkpoints to keep')
    parser.add_argument('--json-log', type=str, default=None,
                       help='Path to JSON log file for metrics tracking')
    
    # Tensor parallelism arguments
    parser.add_argument('--tp-size', type=int, default=1,
                       help='Tensor parallelism size')
    parser.add_argument('--sequence-parallel', action='store_true',
                       help='Enable sequence parallelism')
    
    # Data parallelism arguments
    parser.add_argument('--use-ddp', action='store_true', default=False,
                       help='Use Data Distributed Parallel (DDP) instead of Tensor Parallel')
    parser.add_argument('--parallel-mode', type=str, choices=['tp', 'ddp', 'none', 'auto'], default='auto',
                       help='Parallelism mode: tp (tensor parallel), ddp (data parallel), none, or auto')
    
    # Memory optimization arguments
    parser.add_argument('--use-8bit-adam', action='store_true', default=False,
                       help='Use 8-bit AdamW optimizer')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--use-gradient-monitor', action='store_true', default=True,
                       help='Use advanced gradient monitoring')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False,
                       help='Enable gradient checkpointing to save memory during training')
    parser.add_argument('--gradient-checkpointing-use-reentrant', action='store_true', default=False,
                       help='Use reentrant mode for gradient checkpointing (not recommended for PyTorch 2.0+)')
    
    # Data arguments
    parser.add_argument('--data-dirs', type=str, nargs='+',
                       default=['./data/pile_selected/train'],
                       help='Directories containing parquet files')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Single path to training data (alternative to data-dirs)')
    parser.add_argument('--seq-len', type=int, default=2048,
                       help='Sequence length')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (0 = main process only, auto = system detection)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling')
    
    args = parser.parse_args()
    
    # Validate arguments
    errors = []
    warnings = []
    
    # Model validation
    if args.dim % args.n_heads != 0:
        errors.append(f"Model dimension ({args.dim}) must be divisible by number of heads ({args.n_heads})")
    
    if args.n_activated_experts > args.n_routed_experts:
        errors.append(f"Activated experts ({args.n_activated_experts}) cannot exceed routed experts ({args.n_routed_experts})")
    
    # Data validation
    if args.data_path is None and not args.data_dirs:
        errors.append("Either --data-path or --data-dirs must be specified")
    
    # Worker validation
    if args.num_workers < 0:
        errors.append("Number of workers must be >= 0")
    elif args.num_workers == 0:
        warnings.append("Using num_workers=0 (data loading in main process). This may be slower but more stable.")
    
    # Memory warnings will be shown after config is loaded
    
    # Display errors and warnings
    if errors or warnings:
        if RICH_AVAILABLE:
            console.print("\n[bold]Configuration Issues Found:[/bold]\n")
            
            if errors:
                console.print("[bold red]Errors:[/bold red]")
                for error in errors:
                    console.print(f"  ❌ {error}")
                console.print()
            
            if warnings:
                console.print("[bold yellow]Warnings:[/bold yellow]")
                for warning in warnings:
                    console.print(f"  ⚠️  {warning}")
                console.print()
        else:
            if errors:
                logger.error("Configuration errors:")
                for error in errors:
                    logger.error(f"  - {error}")
            if warnings:
                logger.warning("Configuration warnings:")
                for warning in warnings:
                    logger.warning(f"  - {warning}")
        
        if errors:
            sys.exit(1)
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed(args)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed * rank)
    
    # Setup logging with rich if available
    if RICH_AVAILABLE and rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=True)]
        )
    else:
        # For non-rank-0 processes, log to stderr with higher threshold
        # This prevents interference with rich output on rank 0
        import sys
        logging.basicConfig(
            level=logging.ERROR if rank != 0 else logging.INFO,
            format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stderr if rank != 0 else None
        )
    logger = logging.getLogger(__name__)
    
    if rank == 0:
        # Display welcome message
        if RICH_AVAILABLE:
            welcome_text = """
    🚀 [bold cyan]DeepSeek-V3 Training System[/bold cyan] 🚀
    
    Advanced training with MoE, MLA, and bias-based load balancing
            """
            panel = Panel(
                welcome_text,
                title="[bold]Welcome[/bold]",
                border_style="bright_blue"
            )
            console.print(panel)
            console.print()
        else:
            logger.info("=== DeepSeek-V3 Tensor Parallel Training ===")
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            
    # Validate tensor parallelism
    if args.tp_size > world_size:
        raise ValueError(f"TP size ({args.tp_size}) cannot exceed world size ({world_size})")
    
    if world_size % args.tp_size != 0:
        raise ValueError(f"World size ({world_size}) must be divisible by TP size ({args.tp_size})")
    
    # Load or build model config
    if args.config:
        # Load from JSON file
        logger.info(f"Loading model config from {args.config}")
        with open(args.config, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON config: {e}")
                sys.exit(1)
        logger.info(f"Config mtp_depth: {config.get('mtp_depth', 'NOT FOUND')}")
    else:
        # Build config from command line arguments
        logger.info(f"Building model config from arguments: dim={args.dim}, n_heads={args.n_heads}")
        config = {
            "vocab_size": 128256,  # DeepSeek-V3 vocab size
            "dim": args.dim,
            "inter_dim": int(args.dim * 2.75),  # Standard ratio
            "moe_inter_dim": int(args.dim * 0.6875),  # Standard ratio
            "n_layers": args.n_layers,
            "n_dense_layers": min(2, args.n_layers // 4),  # Reasonable default
            "n_heads": args.n_heads,
            "n_routed_experts": args.n_routed_experts,
            "n_activated_experts": args.n_activated_experts,
            "n_shared_experts": 1,
            "route_scale": 1.0,
            "q_lora_rank": int(args.dim * 0.75),  # Scale with dim
            "kv_lora_rank": int(args.dim * 0.25),  # Scale with dim
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "mscale": 0.707,
            "bias_update_speed": 0.001,
            "initializer_range": 0.02,
            "mtp_depth": 1,
            "mtp_lambda": 0.3,
            "max_seq_len": args.seq_len,
            "max_batch_size": args.batch_size * world_size,
            "gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_use_reentrant": args.gradient_checkpointing_use_reentrant
        }
    
        # Display configuration summary
        if RICH_AVAILABLE:
            table = Table(title="Training Configuration", show_header=True)
            table.add_column("Category", style="cyan")
            table.add_column("Setting", style="magenta") 
            table.add_column("Value", style="green")
            
            # Model settings
            table.add_row("Model", "Dimension", str(args.dim))
            table.add_row("Model", "Layers", str(args.n_layers))
            table.add_row("Model", "Attention Heads", str(args.n_heads))
            table.add_row("Model", "MoE Experts", f"{args.n_activated_experts}/{args.n_routed_experts}")
            
            # Training settings
            if args.parallel_mode == 'ddp':
                effective_batch = args.batch_size * world_size * args.gradient_accumulation_steps
                batch_desc = f"{args.batch_size} × {world_size} GPUs × {args.gradient_accumulation_steps} accum = {effective_batch}"
            else:
                effective_batch = args.batch_size * args.gradient_accumulation_steps
                batch_desc = f"{args.batch_size} × {args.gradient_accumulation_steps} accum = {effective_batch}"
            table.add_row("Training", "Batch Size", batch_desc)
            table.add_row("Training", "Learning Rate", f"{args.learning_rate:.2e}")
            table.add_row("Training", "Max Steps", str(args.max_steps))
            
            # Distributed settings
            table.add_row("Distributed", "World Size", str(world_size))
            table.add_row("Distributed", "Parallel Mode", args.parallel_mode.upper())
            table.add_row("Distributed", "Sequence Parallel", "✓" if args.sequence_parallel else "✗")
            
            # Memory settings
            table.add_row("Memory", "Mixed Precision", "✓" if args.use_amp else "✗")
            table.add_row("Memory", "8-bit Adam", "✓" if args.use_8bit_adam else "✗")
            
            console.print(table)
            console.print()


    # Memory warning based on estimated model size
    estimated_params = estimate_params(config)
    model_size_gb = (estimated_params * 2) / (1024**3)  # bfloat16
    if model_size_gb > 10:
        warnings.append(f"Estimated model size: {model_size_gb:.1f} GB. Consider using gradient checkpointing.")
    
    # Determine parallel mode
    if args.parallel_mode == 'auto':
        # Auto-detect based on model size
        if model_size_gb > 90:  # Model too large for single GPU
            args.parallel_mode = 'tp'
        elif world_size > 1:
            args.parallel_mode = 'ddp'
        else:
            args.parallel_mode = 'none'
    
    # Override with explicit flags
    if args.use_ddp and world_size > 1:
        args.parallel_mode = 'ddp'
    elif args.tp_size > 1:
        args.parallel_mode = 'tp'
    
    logger.info(f"Using parallel mode: {args.parallel_mode}")
    
    # Track model creation time
    model_creation_start = time.time()
    
    # Set default dtype early for model creation
    torch.set_default_dtype(torch.bfloat16)
    
    # Create model based on parallel mode
    # For faster startup, create model directly on GPU with correct dtype
    init_device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    # Set CUDA device for this process to ensure all ops happen on correct GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logger.info(f"Set CUDA device to {local_rank} for faster initialization")
    
    if args.parallel_mode == 'tp':
        # Tensor Parallelism
        tp_config = TPConfig(
            tp_size=args.tp_size,
            sequence_parallel=args.sequence_parallel,
        )
        device_mesh = create_device_mesh(world_size, args.tp_size)
        logger.info(f"Creating model with Tensor Parallelism (TP size: {args.tp_size})")
        with torch.device(init_device):
            model = create_tp_model(config, tp_config, device_mesh)
    else:
        # Data Parallelism or Single GPU
        tp_config = TPConfig(tp_size=1, sequence_parallel=False)
        logger.info("Creating model for DDP or single GPU")
        # Create model directly on GPU to avoid CPU->GPU transfer time
        with torch.device(init_device):
            model = create_tp_model(config, tp_config, None)
    
    # Initialize weights with advanced lottery ticket fix
    logger.info("Initializing model weights with lottery ticket fix...")
    init_start_time = time.time()
    
    def init_weights(module, model_args):
        """
        Advanced weight initialization with lottery ticket fix.
        
        This initialization scheme:
        1. Prevents near-zero weights that can get stuck during training
        2. Uses special handling for Gate modules with sigmoid activation
        3. Initializes expert biases and load tracking properly
        4. Works with both bf16 and fp8 dtypes
        """
        # Standard deviation for initialization
        # Use initializer_range from model args if available
        std = model_args.initializer_range if hasattr(model_args, 'initializer_range') else 0.02
        
        # For smaller models or bfloat16, use smaller std to prevent overflow
        if hasattr(model_args, 'dim') and model_args.dim <= 1024:
            std = min(std, 0.01)  # Smaller std for reduced models
        
        # Minimum magnitude to prevent lottery ticket problem
        # Use 1e-5 for bfloat16 (better precision at small values)
        min_magnitude = 1e-5
        
        # Import custom classes for isinstance check
        from model import Gate, ParallelEmbedding, Linear
        
        # Special handling for Gate modules with bias-based load balancing
        if isinstance(module, Gate):
            # Gate weight matrix for expert routing
            if hasattr(module, 'weight') and module.weight is not None:
                # Use smaller std for sigmoid gates to prevent saturation
                gate_std = 0.005 if module.score_func == "sigmoid" else std
                torch.nn.init.normal_(module.weight, mean=0.0, std=gate_std)
                
                # Post-process to eliminate near-zeros
                with torch.no_grad():
                    # Clip to avoid extreme values
                    module.weight.clamp_(min=-3*gate_std, max=3*gate_std)
                    
                    # Force small values away from zero
                    mask = module.weight.abs() < min_magnitude
                    if mask.any():
                        module.weight[mask] = torch.sign(module.weight[mask]) * min_magnitude
                    
                    # Handle exact zeros
                    zero_mask = module.weight == 0.0
                    if zero_mask.any():
                        module.weight[zero_mask] = torch.empty_like(module.weight[zero_mask]).uniform_(-min_magnitude, min_magnitude)
            
            # Initialize regular bias if present (not expert_biases)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            
            # expert_biases start at zero for unbiased initial routing
            if hasattr(module, 'expert_biases') and module.expert_biases is not None:
                nn.init.zeros_(module.expert_biases)
                
            if rank == 0:  # Only log on main process
                logger.info(f"  Initialized Gate with {module.score_func} activation (std={gate_std if module.score_func == 'sigmoid' else std})")
        
        # Handle ParallelEmbedding
        elif isinstance(module, ParallelEmbedding):
            if hasattr(module, 'weight') and module.weight is not None:
                # Standard embedding initialization
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                
                # Post-processing to eliminate near-zeros
                with torch.no_grad():
                    # Clip to avoid extreme values
                    module.weight.clamp_(min=-3*std, max=3*std)
                    
                    # Skip padding token (usually index 0)
                    if module.weight.shape[0] > 1:
                        # Apply lottery ticket fix to all except padding
                        mask = module.weight[1:].abs() < min_magnitude
                        if mask.any():
                            module.weight[1:][mask] = torch.sign(module.weight[1:][mask]) * min_magnitude
                        
                        # Handle exact zeros (excluding padding token)
                        zero_mask = module.weight[1:] == 0.0
                        if zero_mask.any():
                            module.weight[1:][zero_mask] = torch.empty_like(module.weight[1:][zero_mask]).uniform_(-min_magnitude, min_magnitude)
        
        # Custom and Standard Linear layers
        elif isinstance(module, (Linear, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
            # Post-processing to eliminate near-zeros
            with torch.no_grad():
                # Clip to avoid extreme values
                module.weight.clamp_(min=-3*std, max=3*std)
                
                # Force small values away from zero
                mask = module.weight.abs() < min_magnitude
                if mask.any():
                    module.weight[mask] = torch.sign(module.weight[mask]) * min_magnitude
                
                # Handle exact zeros
                zero_mask = module.weight == 0.0
                if zero_mask.any():
                    module.weight[zero_mask] = torch.empty_like(module.weight[zero_mask]).uniform_(-min_magnitude, min_magnitude)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Standard PyTorch Embedding
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
            # Post-processing to eliminate near-zeros
            with torch.no_grad():
                # Clip to avoid extreme values
                module.weight.clamp_(min=-3*std, max=3*std)
                
                # Skip special tokens (usually indices 0, 1, 2 for DeepSeek-V3)
                if module.weight.shape[0] > 3:
                    # Apply lottery ticket fix to all except special tokens
                    mask = module.weight[3:].abs() < min_magnitude
                    if mask.any():
                        module.weight[3:][mask] = torch.sign(module.weight[3:][mask]) * min_magnitude
                    
                    # Handle exact zeros (excluding special tokens)
                    zero_mask = module.weight[3:] == 0.0
                    if zero_mask.any():
                        module.weight[3:][zero_mask] = torch.empty_like(module.weight[3:][zero_mask]).uniform_(-min_magnitude, min_magnitude)
    
    # Move model to GPU first (if not already there), then initialize
    # This is much faster than initializing on CPU then moving
    if not next(model.parameters()).is_cuda:
        logger.info("Moving model to GPU before initialization for faster startup...")
        model = model.to(device=init_device, dtype=torch.bfloat16)
    else:
        # Model already on GPU, just ensure correct dtype
        model = model.to(dtype=torch.bfloat16)
    
    # Apply initialization to all modules (now on GPU - much faster!)
    logger.info("Initializing weights on GPU for faster startup...")
    model.apply(lambda m: init_weights(m, config))
    
    init_time = time.time() - init_start_time
    logger.info(f"Model initialization completed in {init_time:.2f} seconds")
    
    total_model_setup_time = time.time() - model_creation_start
    logger.info(f"Total model setup time (creation + init + DDP): {total_model_setup_time:.2f} seconds")
    
    # Now wrap with DDP if needed (AFTER model is on correct device and dtype)
    if args.parallel_mode == 'ddp' and world_size > 1:
        logger.info(f"Wrapping model with DistributedDataParallel (world_size: {world_size})")
        # Use gradient_as_bucket_view=True to save memory
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=True  # Required for DeepSeek-V3 model architecture
        )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        logger.info("Gradient checkpointing ENABLED - Trading compute for memory")
        logger.info(f"  Using reentrant mode: {args.gradient_checkpointing_use_reentrant}")
        # Handle DDP wrapper
        actual_model = model.module if hasattr(model, 'module') else model
        if hasattr(actual_model, 'gradient_checkpointing_enable'):
            actual_model.gradient_checkpointing_enable(use_reentrant=args.gradient_checkpointing_use_reentrant)
        else:
            logger.warning("Model does not support gradient checkpointing!")
    else:
        logger.info("Gradient checkpointing DISABLED - Using standard forward/backward")
    
    # Log gradient clipping value and provide recommendations
    logger.info(f"Gradient clipping set to: {args.grad_clip}")
    if args.gradient_accumulation_steps > 1:
        logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size per step: {args.batch_size * args.gradient_accumulation_steps}")
    if args.grad_clip < 5.0:
        logger.warning(f"Gradient clipping value {args.grad_clip} may be too low for training stability.")
        logger.warning("Consider using --grad-clip 10.0 or higher if you see many gradient clipping warnings.")
    
    # Initialize MoE load balancing if model has routed experts
    if config.get('n_routed_experts', 0) > 0:
        moe_utils.reset_gate_statistics(model)
        logger.info("Gate statistics initialized for MoE load balancing")
    
    # Create optimizer
    if args.use_8bit_adam:
        logger.info("Using 8-bit AdamW optimizer")
        # 8-bit Adam requires lr to be a tensor
        lr_tensor = torch.tensor(args.learning_rate)
        optimizer = AdamW8bit(
            model.parameters(),
            lr=lr_tensor,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
    else:
        # Try to create fused optimizer
        optimizer = AdamW(model.parameters(),
                        lr=args.learning_rate,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0.01,
                        fused=True)
        logger.info("Using fused AdamW optimizer")

    # Create gradient monitor with proper model reference
    if args.use_gradient_monitor:
        # Gradient monitor is designed for TP, not DDP
        if args.parallel_mode == 'ddp':
            logger.warning("Gradient monitor is not compatible with DDP mode. Disabling gradient monitor.")
            grad_monitor = None
        else:
            # Single GPU or TP mode
            monitor_model = model
            logger.info("Creating gradient monitor for single GPU/TP mode")
            
            grad_monitor = GradientMonitorTP(
                model=monitor_model,
                base_clip_value=args.grad_clip,
                adaptive_clip=True,
                detailed_monitoring=False,
                tp_size=args.tp_size if args.parallel_mode == 'tp' else 1,
                logger=logger
            )
    else:
        grad_monitor = None
    
    # Create dataset and Determine data path
    if args.data_path:
        data_paths = [args.data_path]
    else:
        data_paths = args.data_dirs
        
    logger.info(f"Using data from: {data_paths}")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3.1-Base",
            trust_remote_code=True
        )
        # Fix pad token
        logger.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
        tokenizer.pad_token_id = 2
        logging.info(f"Tokenizer pad token set to ID: {tokenizer.pad_token_id}")
    except Exception as e:
        # Fatal error if tokenizer cannot be loaded
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)
    # Print estimated model size
    config_with_vocab = config.copy()
    config_with_vocab['vocab_size'] = tokenizer.vocab_size
    estimated_params = estimate_params(config_with_vocab)
    model_size_gb = (estimated_params * 2) / (1024**3)  # bfloat16
    if RICH_AVAILABLE and rank == 0:
        console.print(f"[bold green]Estimated model size:[/bold green] {model_size_gb:.2f} GB")
        console.print(f"[bold green]Total parameters:[/bold green] {estimated_params:,}")
    else:
        logger.info(f"Estimated model size: {model_size_gb:.2f} GB")
        logger.info(f"Total parameters: {estimated_params:,}")

    # Create dataset from parquet files
    from dataset import create_dataloader
    
    # Adjust num_workers based on system capabilities
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    # Check for environment variable override
    if os.environ.get('FORCE_SINGLE_WORKER', '').lower() in ['1', 'true', 'yes']:
        logger.info("FORCE_SINGLE_WORKER is set, using 1 data loading worker")
        effective_num_workers = 1
    elif args.num_workers == 0:
        # User explicitly requested no workers
        effective_num_workers = 0
        logger.info("Using num_workers=0 (data loading in main process)")
    else:
        # PyTorch's suggested max workers (usually cpu_count / num_gpus)
        suggested_workers = max(1, cpu_count // max(torch.cuda.device_count(), 1))
        
        if args.num_workers > suggested_workers:
            logger.warning(f"Reducing num_workers from {args.num_workers} to {suggested_workers} based on system resources")
            logger.info(f"System has {cpu_count} CPUs and {torch.cuda.device_count()} GPUs")
            effective_num_workers = suggested_workers
        else:
            effective_num_workers = args.num_workers
    
    # Further reduce if using DDP to avoid contention
    if args.parallel_mode == 'ddp' and world_size > 1:
        effective_num_workers = max(1, effective_num_workers // world_size)
        logger.info(f"Using {effective_num_workers} workers per process in DDP mode")
    
    # Create the dataloader with distributed support
    dataloader = create_dataloader(
        tokenizer=tokenizer,
        data_dirs=data_paths,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=effective_num_workers,
        rank=rank if args.parallel_mode == 'ddp' else 0,
        world_size=world_size if args.parallel_mode == 'ddp' else 1
    )
    
    # Log memory usage info
    if torch.cuda.is_available() and rank == 0:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"Process memory usage: {mem_info.rss / 1024**3:.2f} GB")
        logger.info(f"DataLoader using pin_memory=True with {effective_num_workers} workers")
        if effective_num_workers > 0:
            estimated_pinned_per_worker = (args.batch_size * args.seq_len * 4) / 1024**3  # 4 bytes per token
            total_pinned_estimate = estimated_pinned_per_worker * effective_num_workers * world_size
            logger.info(f"Estimated pinned memory usage: {total_pinned_estimate:.2f} GB across all processes")
    
    # For DDP with IterableDataset, files are distributed across ranks
    distributed_sampler = None
    if args.parallel_mode == 'ddp' and world_size > 1:
        logger.info(f"Using IterableDataset in DDP mode (rank {rank}/{world_size})")
        logger.info("Files are distributed across ranks for efficient parallel training.")
    
    # Use the dataloader variable name consistently
    dataset = dataloader

    
    # Create scheduler - always use at least LR scheduler for stable training
    scheduler = None
    stable_steps = int(args.max_steps * 0.1) if args.max_steps else 0
    
    if args.n_routed_experts > 0:
        # Use full training scheduler for MoE models
        scheduler = DeepSeekV3TrainingScheduler(
            model=model,
            optimizer=optimizer,
            warmup_steps=args.warmup_steps,
            stable_steps=stable_steps,
            peak_lr=args.learning_rate,
            final_lr=args.learning_rate * 0.1,
            total_steps=args.max_steps,
            mtp_transition_step=int(args.max_steps * 0.7) if args.max_steps else None,
            bias_freeze_step=int(args.max_steps * 0.9) if args.max_steps else None
        )
    else:
        # Use LR scheduler only for non-MoE models
        scheduler = DeepSeekV3LRScheduler(
            optimizer=optimizer,
            warmup_steps=args.warmup_steps,
            stable_steps=stable_steps,
            peak_lr=args.learning_rate,
            final_lr=args.learning_rate * 0.1,
            total_steps=args.max_steps
        )
    
    # Verify scheduler was created
    if scheduler is None:
        logger.warning("No learning rate scheduler created! This may lead to training instability.")
        logger.warning("Training will use a constant learning rate, which is not recommended.")
    else:
        logger.info(f"Learning rate scheduler initialized: warmup={args.warmup_steps}, peak_lr={args.learning_rate}")
        if isinstance(scheduler, DeepSeekV3LRScheduler):
            logger.info(f"Scheduler details: stable_steps={scheduler.stable_steps}, total_steps={scheduler.total_steps}")
            logger.info(f"Initial LR at step 0: {scheduler.get_lr():.6f} (min warmup LR = {args.learning_rate * 0.1:.6f})")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(args.save_dir, args.keep_checkpoints, rank)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    global_step = 0
    if args.resume:
        global_step, start_epoch = resume_training(args, model, optimizer, scheduler)
    
    # Training state
    nan_count = 0
    max_nan_steps = 10  # Allow up to 10 NaN steps before stopping
    
    # Initialize JSON logging
    json_log_path = None
    json_metrics = []
    if args.json_log and rank == 0:
        json_log_path = args.json_log
        # Create directory if needed
        os.makedirs(os.path.dirname(json_log_path) if os.path.dirname(json_log_path) else '.', exist_ok=True)
        logger.info(f"JSON metrics will be saved to: {json_log_path}")
    
    # CUDA warmup to prevent lazy loading hangs
    if torch.cuda.is_available():
        logger.info("Performing CUDA warmup...")
        # Create dummy tensors to force CUDA initialization
        dummy = torch.randn(10, 10, device='cuda', dtype=torch.float16)
        dummy2 = torch.randn(10, 10, device='cuda', dtype=torch.float16)
        _ = dummy @ dummy2  # Force CUDA kernel compilation
        
        # Force optimizer CUDA initialization if using 8-bit Adam
        if args.use_8bit_adam:
            # Create a dummy parameter and run optimizer step to initialize CUDA kernels
            dummy_param = torch.nn.Parameter(torch.randn(100, 100, device='cuda', dtype=torch.float16))
            dummy_optimizer = type(optimizer)([dummy_param], lr=1e-3)
            dummy_param.grad = torch.randn_like(dummy_param)
            dummy_optimizer.step()
            del dummy_optimizer, dummy_param
        
        torch.cuda.synchronize()
        logger.info("CUDA warmup complete")
    
    # Training loop with progress tracking
    logger.info("Starting training...")
    model.train()
    
    # Debug: Check model parameters require gradients
    actual_model = model.module if hasattr(model, 'module') else model
    total_params = sum(1 for p in actual_model.parameters())
    trainable_params = sum(1 for p in actual_model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params} total parameters, {trainable_params} require gradients")
    if trainable_params == 0:
        logger.error("ERROR: No model parameters require gradients! Training will not work.")
        raise RuntimeError("Model has no trainable parameters")
    
    # Create progress bar if using rich
    progress = None
    progress_task = None
    if RICH_AVAILABLE and rank == 0:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        )
        progress.start()
        progress_task = progress.add_task(
            "[cyan]|", 
            total=args.max_steps
        )
    
    # Track metrics
    current_lr = args.learning_rate  # Initialize current learning rate
    
    # Timing variables for tokens/sec calculation
    step_start_time = time.time()
    training_start_time = time.time()
    total_tokens_processed = 0
    last_log_step = global_step
    
    # Initialize grad_norm to avoid undefined variable in logging
    grad_norm_value = 0.0
    
    for epoch in range(start_epoch, args.num_epochs):
        # For IterableDataset with DDP, we rely on random seeds for different data per rank
        if args.parallel_mode == 'ddp' and world_size > 1:
            # Set a different seed for each epoch and rank to ensure variety
            torch.manual_seed(args.seed + epoch * 1000 + rank)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed + epoch * 1000 + rank)
        
        # Skip to the right position if resuming mid-epoch
        if epoch == start_epoch and global_step > 0:
            steps_to_skip = global_step % (len(dataset) // args.gradient_accumulation_steps)
            if steps_to_skip > 0:
                logger.info(f"Skipping {steps_to_skip} steps to resume from step {global_step}")
        
        # Create data iterator for this epoch
        data_iter = iter(dataset)
        
        for batch_idx, batch in enumerate(data_iter):
            logger.info(f"Processing batch {batch_idx + 1} (global step {global_step})")
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).cuda()
            labels = batch.get('labels', input_ids.clone()).cuda()
            
            # Debug: Check if any labels are out of vocab range
            if global_step == 0 and batch_idx == 0:
                vocab_size = config.get('vocab_size', 128256)
                max_label = labels.max().item()
                min_label = labels[labels != -100].min().item() if (labels != -100).any() else -1
                logger.info(f"[DEBUG] Vocab size: {vocab_size}")
                logger.info(f"[DEBUG] Labels range: min={min_label}, max={max_label}")
                logger.info(f"[DEBUG] Input IDs range: min={input_ids.min().item()}, max={input_ids.max().item()}")
                if max_label >= vocab_size:
                    logger.error(f"[ERROR] Labels contain values >= vocab_size! max={max_label} >= {vocab_size}")
                    invalid_mask = labels >= vocab_size
                    invalid_count = invalid_mask.sum().item()
                    logger.error(f"[ERROR] Found {invalid_count} invalid labels out of {labels.numel()}")
            
            # Check if we should zero gradients (first batch in accumulation)
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
                # Debug model parameters before forward pass
                if global_step == 0 and batch_idx == 0:
                    actual_model = model.module if hasattr(model, 'module') else model
                    param_info = []
                    for name, param in list(actual_model.named_parameters())[:5]:  # First 5 params
                        param_info.append(f"{name}: requires_grad={param.requires_grad}, dtype={param.dtype}")
                    logger.info(f"[DEBUG] First 5 model parameters:\n" + "\n".join(param_info))
                    
                    # Check model's actual vocab size
                    if hasattr(actual_model, 'head'):
                        head_out_features = actual_model.head.weight.shape[0]
                        logger.info(f"[DEBUG] Model head output features (vocab size): {head_out_features}")
                    if hasattr(actual_model, 'embed'):
                        embed_vocab_size = actual_model.embed.vocab_size
                        logger.info(f"[DEBUG] Model embedding vocab size: {embed_vocab_size}")            # Ensure gradients are enabled
            torch.set_grad_enabled(True)
            
            # Forward pass with mixed precision (use bfloat16 to match model dtype)
            # DEBUGGING: Completely disable autocast to see if it's causing gradient issues
            use_autocast = False  # args.use_amp and global_step > 5
            if global_step == 0 and batch_idx == 0:
                logger.info(f"[DEBUG] Autocast is DISABLED for debugging")
            with torch.amp.autocast('cuda', enabled=use_autocast, dtype=torch.bfloat16):
                # Forward pass with attention mask for proper padding handling
                if config.get('mtp_depth', 0) > 0:
                    try:
                        # MTP enabled, return logits and mtp_logits
                        logger.info("Using MTP (Multi-Task Prediction) for auxiliary loss")
                        # Handle DDP wrapper
                        actual_model = model.module if hasattr(model, 'module') else model
                        if hasattr(actual_model, 'mtp_lambda'):
                            logger.info(f"Using MTP lambda: {actual_model.mtp_lambda}")

                        logits, mtp_logits = model(tokens=input_ids, return_mtp=True, attention_mask=
                                                   attention_mask)
                    except Exception as e:
                        logger.error(f"Forward pass failed at step {global_step}, batch {batch_idx} with error: {e}\n{torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                        raise e
                else:
                    # Debug: Check if model call is breaking gradients
                    if global_step == 0 and batch_idx == 0:
                        # Check a parameter before and after forward
                        first_param = next(model.parameters())
                        logger.info(f"[DEBUG] Before forward: first param requires_grad={first_param.requires_grad}")
                        logger.info(f"[DEBUG] Before forward: first param is_leaf={first_param.is_leaf}")
                    
                    logits = model(tokens=input_ids, attention_mask=attention_mask)
                    mtp_logits = None
                    
                    if global_step == 0 and batch_idx == 0:
                        logger.info(f"[DEBUG] After forward: first param requires_grad={first_param.requires_grad}")
                        logger.info(f"[DEBUG] After forward: logits is_leaf={logits.is_leaf}")
                        logger.info(f"[DEBUG] After forward: logits grad_fn={logits.grad_fn}")
                        logger.info(f"[DEBUG] After forward: logits shape={logits.shape}")
                        logger.info(f"[DEBUG] After forward: world_size={dist.get_world_size() if dist.is_initialized() else 1}")
                
                # Debug logits gradient requirement
                if global_step == 0 and batch_idx < 4:
                    logger.info(f"[DEBUG] Batch {batch_idx}: logits requires_grad: {logits.requires_grad}")
                    logger.info(f"[DEBUG] logits dtype: {logits.dtype}")
                    logger.info(f"[DEBUG] input_ids requires_grad: {input_ids.requires_grad}")
                    # Check if model is in training mode
                    actual_model = model.module if hasattr(model, 'module') else model
                    logger.info(f"[DEBUG] Model training mode: {actual_model.training}")
                    logger.info(f"[DEBUG] Autocast enabled: {args.use_amp}")
                    logger.info(f"[DEBUG] torch.is_grad_enabled(): {torch.is_grad_enabled()}")
                
                # Calculate main loss
                main_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100   # Ignore padding tokens in loss calculation
                )
                
                # Debug loss gradient requirement
                if global_step == 0 and batch_idx < 4:
                    logger.info(f"[DEBUG] Batch {batch_idx}: main_loss requires_grad AFTER computation: {main_loss.requires_grad}")
                
                # Calculate MTP loss if enabled
                if mtp_logits is not None:
                    # Handle DDP wrapper
                    actual_model = model.module if hasattr(model, 'module') else model
                    current_mtp_lambda = getattr(actual_model, 'mtp_lambda', config.get('mtp_lambda', 0.3))
                    mtp_loss = compute_mtp_loss(mtp_logits, labels, current_mtp_lambda)
                else:
                    # Create zero tensor with same device and dtype
                    mtp_loss = torch.tensor(0.0, device=input_ids.device, dtype=main_loss.dtype, requires_grad=True)
                
                # For DDP, ensure losses are properly reduced across ranks if needed
                if args.parallel_mode == 'ddp' and world_size > 1:
                    # DDP already averages gradients, but we need to ensure loss values are consistent
                    # This is important for logging and monitoring
                    main_loss = main_loss.mean() if main_loss.dim() > 0 else main_loss
                    if isinstance(mtp_loss, torch.Tensor) and mtp_loss.dim() > 0:
                        mtp_loss = mtp_loss.mean()
                
                # Combine losses and scale by gradient accumulation steps
                total_loss = (main_loss + mtp_loss) / args.gradient_accumulation_steps
                
                # Debug gradient tracking in DDP
                if args.parallel_mode == 'ddp' and global_step == 0 and batch_idx == 0:
                    logger.info(f"[DEBUG] main_loss requires_grad: {main_loss.requires_grad}")
                    logger.info(f"[DEBUG] mtp_loss type: {type(mtp_loss)}, requires_grad: {mtp_loss.requires_grad if isinstance(mtp_loss, torch.Tensor) else 'N/A'}")
                    logger.info(f"[DEBUG] total_loss requires_grad: {total_loss.requires_grad}")

            # Backward pass
            try:
                total_loss.backward()
            except Exception as e:
                logger.error(f"Backward pass failed at step {global_step}, batch {batch_idx} with error: {e}")
                raise e
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # Gradient synchronization based on parallel mode
                if args.parallel_mode == 'tp' and args.tp_size > 1:
                    # All-reduce gradients across TP ranks
                    all_reduce_gradients(model, args.tp_size)
                # Note: DDP automatically synchronizes gradients during backward pass
                
                # Gradient clipping and monitoring
                current_grad_norm = 0.0
                if grad_monitor is not None and args.parallel_mode != 'ddp':
                    # Use gradient monitor only for TP mode, not DDP
                    current_grad_norm, grad_stats = grad_monitor.clip_and_monitor(
                        max_norm=args.grad_clip,
                    )
                    # Extract the actual gradient norm value
                    current_grad_norm = grad_stats.get('grad_norm', current_grad_norm)
                else:
                    # Manual gradient clipping without monitoring
                    # For DDP, we need to use the correct model reference
                    if args.parallel_mode == 'ddp' and hasattr(model, 'module'):
                        clip_model = model.module
                    else:
                        clip_model = model
                    
                    # Debug: manually compute gradient norm before clipping
                    if global_step < 5:
                        total_norm = 0.0
                        param_count = 0
                        params_without_grad = 0
                        params_not_requiring_grad = 0
                        for p in clip_model.parameters():
                            if not p.requires_grad:
                                params_not_requiring_grad += 1
                            elif p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                param_count += 1
                            else:
                                params_without_grad += 1
                        total_norm = total_norm ** 0.5
                        logger.info(f"[DEBUG] Manual grad norm calculation: {total_norm:.6f} from {param_count} params with gradients")
                        logger.info(f"[DEBUG] Params without grad: {params_without_grad}, Params not requiring grad: {params_not_requiring_grad}")
                        logger.info(f"[DEBUG] Total params: {sum(1 for _ in clip_model.parameters())}")
                    
                    current_grad_norm = torch.nn.utils.clip_grad_norm_(clip_model.parameters(), args.grad_clip)
                    current_grad_norm = current_grad_norm.item() if isinstance(current_grad_norm, torch.Tensor) else current_grad_norm
                    
                    # Debug logging for gradient norm
                    if global_step < 10 or global_step % 100 == 0:
                        logger.info(f"[DEBUG] Step {global_step}: Pre-sync grad_norm = {current_grad_norm:.6f}")
                    
                    # For DDP with multiple GPUs, we need to ensure gradient norm is computed correctly
                    # DDP already synchronizes gradients, but gradient norm computation happens locally
                    if args.parallel_mode == 'ddp' and world_size > 1:
                        # Compute gradient norm across all ranks
                        grad_norm_tensor = torch.tensor([current_grad_norm], device=f'cuda:{local_rank}')
                        torch.distributed.all_reduce(grad_norm_tensor, op=torch.distributed.ReduceOp.MAX)
                        current_grad_norm = grad_norm_tensor.item()
                        
                        if global_step < 10 or global_step % 100 == 0:
                            logger.info(f"[DEBUG] Step {global_step}: Post-sync grad_norm = {current_grad_norm:.6f}")
                    
                    grad_stats = {'grad_norm': current_grad_norm}
                
                # Update the global grad_norm_value for logging
                grad_norm_value = current_grad_norm
                
                # Always log for first few steps to debug
                if global_step < 10:
                    logger.info(f"[DEBUG] Step {global_step}: grad_norm_value set to {grad_norm_value:.6f}")
                    logger.info(f"[DEBUG] batch_idx={batch_idx}, gradient_accumulation_steps={args.gradient_accumulation_steps}")
                    logger.info(f"[DEBUG] Is optimizer step: {(batch_idx + 1) % args.gradient_accumulation_steps == 0}")
                
                # Ensure all CUDA operations are complete before optimizer step
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Optimizer step with timeout protection
                try:
                    logger.info(f"Optimizer step for global step {global_step}, batch {batch_idx + 1}")
                    optimizer.step()
                except Exception as e:
                    logger.error(f"Optimizer step failed at global_step {global_step}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Error message: {str(e)}")
                    if "ldconfig" in str(e).lower() or "cuda" in str(e).lower():
                        logger.error("This appears to be a CUDA initialization issue.")
                        logger.error("Try setting CUDA_MODULE_LOADING=EAGER or running with CUDA_LAUNCH_BLOCKING=1")
                    raise e
                
                # Scheduler step (only after optimizer step)
                current_lr = args.learning_rate
                if scheduler is not None:
                    # For DDP, ensure scheduler state is synchronized across ranks
                    if args.parallel_mode == 'ddp' and world_size > 1:
                        # Synchronize the scheduler's current step across ranks
                        # This ensures all ranks have the same view of training progress
                        if hasattr(scheduler, 'current_step'):
                            step_tensor = torch.tensor([scheduler.current_step], dtype=torch.long, device=f'cuda:{local_rank}')
                            torch.distributed.all_reduce(step_tensor, op=torch.distributed.ReduceOp.MAX)
                            expected_step = step_tensor.item()
                            
                            if scheduler.current_step != expected_step:
                                logger.warning(f"Scheduler step mismatch on rank {rank}: local={scheduler.current_step}, expected={expected_step}. Synchronizing.")
                                scheduler.current_step = expected_step
                    
                    if isinstance(scheduler, DeepSeekV3LRScheduler):
                        # LR scheduler returns float and updates optimizer internally
                        if global_step < 10 and rank == 0:
                            logger.info(f"[DEBUG] Before scheduler.step(): current_step={scheduler.current_step}, lr={scheduler.get_lr():.6f}")
                        current_lr = scheduler.step()
                        if global_step < 10 and rank == 0:
                            logger.info(f"[DEBUG] After scheduler.step(): current_step={scheduler.current_step}, lr={current_lr:.6f}")
                    else:
                        # Training scheduler returns dict
                        scheduler_info = scheduler.step()
                        current_lr = scheduler_info.get('lr', args.learning_rate)
                    
                    # For DDP, broadcast the learning rate from rank 0 to ensure consistency
                    if args.parallel_mode == 'ddp' and world_size > 1:
                        lr_tensor = torch.tensor([current_lr], dtype=torch.float32, device=f'cuda:{local_rank}')
                        torch.distributed.broadcast(lr_tensor, src=0)
                        current_lr = lr_tensor.item()
                
                # Verify optimizer LR was updated (for debugging)
                actual_lr = optimizer.param_groups[0]['lr']
                if isinstance(actual_lr, torch.Tensor):
                    actual_lr = actual_lr.item()
                
                # Verify LR update was successful
                if abs(actual_lr - current_lr) > 1e-10:
                    # With proper synchronization, this should rarely happen
                    logger.debug(f"LR update verification: scheduler={current_lr:.6e}, optimizer={actual_lr:.6e}, diff={abs(actual_lr - current_lr):.2e}")
                    
                    # Only force update if difference is significant
                    if abs(actual_lr - current_lr) / max(actual_lr, current_lr, 1e-10) > 0.01:  # 1% tolerance
                        logger.warning(f"Significant LR mismatch: forcing optimizer update")
                        for param_group in optimizer.param_groups:
                            if isinstance(param_group['lr'], torch.Tensor):
                                param_group['lr'].fill_(current_lr)
                            else:
                                param_group['lr'] = current_lr
                
                # Adjust expert biases based on load statistics
                if args.n_routed_experts > 0:
                    moe_utils.adjust_all_gate_biases(model)
                
                # Increment global step (only after optimizer step)
                global_step += 1
            
            # Only process metrics and logging after gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # Create outputs dictionary for logging
                outputs = {
                    'loss': main_loss.detach().item(),
                    'aux_loss': mtp_loss.detach().item() if isinstance(mtp_loss, torch.Tensor) else 0.0,
                    'total_loss': total_loss.detach().item() * args.gradient_accumulation_steps  # Unscale for logging
                }
                
                # Check for NaN
                if torch.isnan(torch.tensor(outputs['loss'])):
                    nan_count += 1
                    if rank == 0:
                        logger.warning(f"Step {global_step}: NaN detected! (count: {nan_count}/{max_nan_steps})")
                        if nan_count == 1:
                            logger.info("Debug info:")
                            logger.info(f"  Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
                            logger.info(f"  Learning rate: {current_lr}")
                            logger.info(f"  Batch size: {args.batch_size}, Seq length: {args.seq_len}")
                            logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
                            logger.info("  This often indicates initialization issues or learning rate too high")
                    
                    if nan_count >= max_nan_steps:
                        logger.error(f"Too many NaN steps ({nan_count}), stopping training")
                        break
                    continue
                else:
                    nan_count = 0  # Reset on successful step
                
                # Logging
                if global_step % args.log_interval == 0 and rank == 0:
                    # grad_norm_value is already updated from gradient clipping above
                    # For gradient accumulation steps, we may not have updated grad norm yet
                    is_optimizer_step = (batch_idx + 1) % args.gradient_accumulation_steps == 0
                    
                    metrics = {
                        'loss': outputs['loss'],
                        'learning_rate': current_lr,
                        'grad_norm': grad_norm_value if is_optimizer_step else grad_norm_value,  # Show last value during accumulation
                        'global_step': global_step,
                    }
                    
                    # Debug logging for grad norm issue
                    if global_step < 10:
                        logger.info(f"[DEBUG] Metrics at step {global_step}: grad_norm={metrics['grad_norm']:.6f}, is_optimizer_step={is_optimizer_step}")
                    
                    # Add MoE metrics if available
                    if 'aux_loss' in outputs:
                        metrics['aux_loss'] = outputs['aux_loss']
                        
                    # Add gradient monitor stats
                    if grad_monitor is not None and 'grad_stats' in locals():
                        metrics['grad_clip_rate'] = grad_stats.get('clip_rate', 0)
                    
                    # Calculate tokens per second
                    current_time = time.time()
                    time_elapsed = current_time - step_start_time
                    
                    # Calculate instantaneous tokens/sec (since last log)
                    if time_elapsed > 0 and global_step > last_log_step:
                        steps_since_log = global_step - last_log_step
                        # For DDP, each GPU processes different data. For TP, they process the same data
                        if args.parallel_mode == 'ddp':
                            tokens_this_log = args.batch_size * args.seq_len * args.gradient_accumulation_steps * steps_since_log * world_size
                        else:
                            tokens_this_log = args.batch_size * args.seq_len * args.gradient_accumulation_steps * steps_since_log
                        tokens_per_sec = tokens_this_log / time_elapsed
                        metrics['tokens/sec'] = int(tokens_per_sec)
                        
                        # Calculate average tokens/sec (since training start)
                        total_time = current_time - training_start_time
                        total_tokens_processed += tokens_this_log
                        avg_tokens_per_sec = total_tokens_processed / total_time if total_time > 0 else 0
                        metrics['avg_tokens/sec'] = int(avg_tokens_per_sec)
                        
                        # Update timing for next interval
                        step_start_time = current_time
                        last_log_step = global_step
                    
                    # Add timestamp and epoch info
                    metrics['timestamp'] = current_time
                    metrics['epoch'] = epoch + 1
                    metrics['total_loss'] = outputs.get('total_loss', outputs['loss'])
                    
                    # Add MTP lambda if using MTP
                    # Handle DDP wrapper
                    actual_model = model.module if hasattr(model, 'module') else model
                    if hasattr(actual_model, 'mtp_lambda'):
                        metrics['mtp_lambda'] = getattr(actual_model, 'mtp_lambda', config.get('mtp_lambda', 0.3))
                    
                    # JSON logging
                    if json_log_path:
                        json_metrics.append(metrics.copy())
                        # Write to file (overwrite each time for crash safety)
                        try:
                            with open(json_log_path, 'w') as f:
                                json.dump({
                                    'config': config,
                                    'args': vars(args),
                                    'metrics': json_metrics
                                }, f, indent=2)
                        except Exception as e:
                            logger.warning(f"Failed to write JSON log: {e}")
                    
                    # Enhanced logging with rich
                    if RICH_AVAILABLE:
                        # Create a formatted string with color
                        log_parts = [f"[bold green]Step {global_step}[/bold green]"]
                        log_parts.append(f"Loss: [yellow]{metrics['loss']:.4f}[/yellow]")
                        log_parts.append(f"LR: [cyan]{metrics['learning_rate']:.2e}[/cyan]")
                        log_parts.append(f"Grad: [magenta]{metrics['grad_norm']:.2f}[/magenta]")
                        
                        if 'tokens/sec' in metrics:
                            log_parts.append(f"Speed: [blue]{metrics['tokens/sec']:,} tok/s[/blue]")
                        
                        if 'aux_loss' in metrics and metrics['aux_loss'] > 0:
                            log_parts.append(f"Aux: {metrics['aux_loss']:.4f}")
                        # Calculate time remaining
                        steps_remaining = args.max_steps - global_step
                        if 'tokens/sec' in metrics and metrics['tokens/sec'] > 0:
                            # Calculate based on tokens per second
                            if args.parallel_mode == 'ddp':
                                tokens_per_step = args.batch_size * args.seq_len * args.gradient_accumulation_steps * world_size
                            else:
                                tokens_per_step = args.batch_size * args.seq_len * args.gradient_accumulation_steps
                            seconds_per_step = tokens_per_step / metrics['tokens/sec']
                            remaining_seconds = steps_remaining * seconds_per_step
                            
                            # Format time remaining
                            hours = int(remaining_seconds // 3600)
                            minutes = int((remaining_seconds % 3600) // 60)
                            if hours > 0:
                                time_str = f"{hours}h {minutes}m"
                            else:
                                minutes = int(remaining_seconds // 60)
                                seconds = int(remaining_seconds % 60)
                                time_str = f"{minutes}m {seconds}s"
                            
                            log_parts.append(f"ETA: [green]{time_str}[/green]")
                        
                        progress.update(
                            progress_task,
                            advance=args.log_interval,
                            description=f" | ".join(log_parts),
                        )
                    else:
                        logger.info(f"Step {global_step}: " + 
                                    ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                            for k, v in metrics.items()]))
                
                # Save checkpoint
                if global_step % args.save_interval == 0 and rank == 0:
                    # Get state dict from DDP module if using DDP
                    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    checkpoint_dict = {
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'global_step': global_step,
                        'epoch': epoch,
                        'config': config,
                        'args': vars(args),
                    }
                    
                    # Include current metrics for checkpoint selection
                    checkpoint_metrics = {
                        'loss': outputs['loss'],
                        'learning_rate': current_lr,
                        'grad_norm': grad_norm_value if 'grad_norm_value' in locals() else 0.0
                    }
                    
                    checkpoint_manager.save_checkpoint(checkpoint_dict, global_step, checkpoint_metrics)
            
            # Check if we've reached max steps
            if global_step >= args.max_steps:
                logger.info(f"Reached maximum steps ({args.max_steps}), stopping training")
                break
        
        if global_step >= args.max_steps:
            break
        
        # End of epoch logging
        if rank == 0:
            logger.info(f"Completed epoch {epoch + 1}/{args.num_epochs}")
            if args.n_routed_experts > 0:
                stats = moe_utils.get_load_balance_metrics(model)
                logger.info("Load balance statistics:")
                for gate_name, gate_stats in stats.items():
                    if 'load_variance' in gate_stats:
                        logger.info(f"  {gate_name}: Load Var = {gate_stats['load_variance']:.6f}")
    
    # Final JSON log write
    if json_log_path and rank == 0:
        try:
            # Add final summary to JSON
            final_summary = {
                'final_step': global_step,
                'final_loss': outputs.get('loss', 'N/A') if 'outputs' in locals() else 'N/A',
                'final_lr': current_lr,
                'total_epochs': epoch + 1 if 'epoch' in locals() else 0,
                'completed': True
            }
            
            with open(json_log_path, 'w') as f:
                json.dump({
                    'config': config,
                    'args': vars(args),
                    'metrics': json_metrics,
                    'summary': final_summary
                }, f, indent=2)
            logger.info(f"Final metrics saved to: {json_log_path}")
        except Exception as e:
            logger.warning(f"Failed to write final JSON log: {e}")
    
    # Final summary
    if rank == 0:
        logger.info("\n" + "="*70)
        logger.info("Training Summary")
        logger.info("="*70)
        logger.info(f"Total steps completed: {global_step}")
        logger.info(f"Final learning rate: {current_lr:.2e}")
        logger.info(f"Configuration:")
        logger.info(f"  Model dimension: {config['dim']}")
        logger.info(f"  Number of layers: {config['n_layers']}")
        logger.info(f"  Number of experts: {config.get('n_routed_experts', 'N/A')}")
        logger.info(f"  Parallel mode: {args.parallel_mode.upper()}")
        if args.parallel_mode == 'tp':
            logger.info(f"  Tensor parallel size: {args.tp_size}")
        logger.info(f"  Batch size per GPU: {args.batch_size}")
        logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        if args.parallel_mode == 'ddp':
            logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
        else:
            logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        
        if grad_monitor:
            summary = grad_monitor.get_summary()
            logger.info(f"\nGradient statistics:")
            logger.info(f"  Clip rate: {summary['clip_rate']:.2%}")
            logger.info(f"  Average gradient norm: {summary.get('grad_norm_stats', {}).get('mean', 'N/A'):.4f}")
        
        logger.info("="*70)
    
    # Save final checkpoint
    if rank == 0:
        logger.info("Saving final checkpoint...")
        # Get state dict from DDP module if using DDP
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        final_checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'global_step': global_step,
            'epoch': epoch if 'epoch' in locals() else 0,
            'config': config,
            'args': vars(args),
        }
        
        final_metrics = {
            'loss': outputs.get('loss', float('inf')) if 'outputs' in locals() else float('inf'),
            'final': True
        }
        
        checkpoint_manager.save_checkpoint(final_checkpoint, global_step, final_metrics)
        
        # Also save as 'final.pt' for easy identification
        final_path = os.path.join(args.save_dir, 'final.pt')
        torch.save(final_checkpoint, final_path)
        logger.info(f"Saved final checkpoint to {final_path}")
    
    # Cleanup
    # Clean up progress bar
    if progress is not None:
        progress.stop()
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # Display completion message
    if rank == 0:
        if RICH_AVAILABLE:
            console.print(f"\n[bold green]✅ Training completed successfully![/bold green]")
            console.print(f"Total steps: {global_step}")
            console.print(f"Final loss: {outputs.get('loss', 'N/A'):.4f}" if 'outputs' in locals() else "")
        else:
            logger.info("Training completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Training interrupted by user[/yellow]")
        else:
            print("\nTraining interrupted by user")
        sys.exit(0)
    except torch.cuda.OutOfMemoryError as e:
        if RICH_AVAILABLE:
            console.print(f"\n[bold red]❌ CUDA Out of Memory Error[/bold red]")
            console.print("\n[yellow]Suggestions:[/yellow]")
            console.print("  • Reduce batch size")
            console.print("  • Enable gradient checkpointing")
            console.print("  • Use more gradient accumulation steps")
            console.print("  • Enable 8-bit Adam optimizer (--use-8bit-adam)")
            console.print("  • Use tensor parallelism across more GPUs")
        else:
            print(f"\nCUDA Out of Memory: {str(e)}")
            print("Try reducing batch size or enabling memory optimizations")
        sys.exit(1)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[bold red]❌ Training failed with error:[/bold red]")
            console.print_exception()
        else:
            print(f"\nTraining failed: {str(e)}")
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore stdout for non-rank-0 processes
        if not _is_rank_zero and '_original_stdout' in globals():
            sys.stdout.close()
            sys.stdout = _original_stdout
