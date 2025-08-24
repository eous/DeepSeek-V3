import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

# Global variables for tensor parallelism
# These should only be set > 1 for actual tensor parallelism (model sharding)
# For DDP, these should remain 1 since each rank has the full model
world_size = 1
rank = 0

def set_tensor_parallel_config(tp_world_size: int, tp_rank: int):
    """Set tensor parallel configuration. Only call this for actual TP, not DDP."""
    global world_size, rank
    world_size = tp_world_size
    rank = tp_rank

# Optional kernel imports for FP8 optimization
try:
    from inference.kernel import act_quant, weight_dequant, fp8_gemm
    HAS_FP8_KERNEL = True
except ImportError:
    try:
        from inference.kernel import act_quant, weight_dequant, fp8_gemm
        HAS_FP8_KERNEL = True
    except ImportError:
        HAS_FP8_KERNEL = False
        # Define dummy functions to maintain compatibility
        def act_quant(*args, **kwargs):
            return args[0] if args else None
        def weight_dequant(*args, **kwargs):
            return args[0] if args else None
        def fp8_gemm(a, b, *args, **kwargs):
            return torch.matmul(a, b)


block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    # load balancing
    bias_update_speed: float = 0.001
    # weight initialization
    initializer_range: float = 0.02
    # multi-token prediction
    mtp_depth: int = 1  # Number of additional tokens to predict (D=1 for DeepSeek-V3)
    mtp_lambda: float = 0.3  # MTP loss weight (0.3 for first 10T tokens, then 0.1)
    # gradient checkpointing
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    gradient_checkpointing_use_reentrant: bool = False  # Use non-reentrant mode (PyTorch 2.0+)


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    # Always return cos/sin to avoid complex number issues with gradient checkpointing
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    freqs_cis = torch.stack([freqs_cos, freqs_sin], dim=-1)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    
    # Check if freqs_cis is complex or real
    if torch.is_complex(freqs_cis):
        # Original implementation for complex tensors
        x = x.float()
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        x_rotated = x * freqs_cis
        y = torch.view_as_real(x_rotated).flatten(3)
        return y.type(dtype)
    else:
        # Alternative implementation for real tensors (gradient checkpointing case)
        # freqs_cis should already contain cos and sin values
        x = x.float()
        x_shape = x.shape
        x = x.reshape(*x_shape[:-1], -1, 2)  # [..., head_dim/2, 2]
        
        # Assuming freqs_cis is [seq_len, head_dim/2, 2] with cos and sin
        if freqs_cis.dim() == 2:
            # Need to expand freqs_cis to have the right shape
            seq_len = x_shape[1]
            freqs_cis = freqs_cis[:seq_len].view(seq_len, -1, 2)
        
        freqs_cis = freqs_cis.view(1, x_shape[1], 1, x.shape[-2], 2)
        cos = freqs_cis[..., 0]
        sin = freqs_cis[..., 1]
        
        # Apply rotation using real arithmetic
        x_real = x[..., 0]
        x_imag = x[..., 1]
        
        # Rotation: [cos, -sin; sin, cos] * [x_real; x_imag]
        x_out_real = x_real * cos - x_imag * sin
        x_out_imag = x_real * sin + x_imag * cos
        
        # Stack and reshape back
        x_out = torch.stack([x_out_real, x_out_imag], dim=-1)
        x_out = x_out.flatten(-2)
        
        return x_out.type(dtype)


def compute_mtp_loss(mtp_predictions: list, targets: torch.Tensor, 
                     mtp_lambda: float = 0.3, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute Multi-Token Prediction loss according to DeepSeek-V3 paper.
    
    L_MTP^k = CrossEntropy(P_{2+k:T+1}^k, t_{2+k:T+1})    (Eq. 24)
    L_MTP = (λ/D) * Σ(k=1 to D) L_MTP^k                   (Eq. 25)
    
    Args:
        mtp_predictions: List of predictions for each depth
        targets: Target tokens [batch, seq_len]
        mtp_lambda: Weight for MTP loss (λ)
        ignore_index: Index to ignore in loss computation (for padding)
        
    Returns:
        Weighted MTP loss
    """
    losses = []
    
    for depth, pred in enumerate(mtp_predictions):
        # Offset targets by depth+2 (since depth 0 predicts position i+2)
        # This implements the offset in Eq. 24: P_{2+k:T+1}^k vs t_{2+k:T+1}
        if depth + 2 < targets.size(1):
            target_slice = targets[:, depth+2:]
            pred_slice = pred[:, :target_slice.size(1)]
            
            # Compute cross entropy loss
            loss = F.cross_entropy(
                pred_slice.reshape(-1, pred_slice.size(-1)),
                target_slice.reshape(-1),
                ignore_index=ignore_index
            )
            losses.append(loss)
    
    # Average over depths and apply weight (Eq. 25)
    if losses:
        # Use torch.stack and mean to avoid multiple graph traversals
        total_loss = torch.stack(losses).mean()
        return mtp_lambda * total_loss
    return torch.tensor(0.0, device=targets.device)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # Check if we're in training mode - if so, don't use KV cache
        if self.training and seqlen > 1:
            # Training mode: compute attention without caching
            if attn_impl == "naive":
                q = torch.cat([q_nope, q_pe], dim=-1)
                kv = self.wkv_b(self.kv_norm(kv))
                kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
                scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
            else:
                wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
                wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
                q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
                kv_norm = self.kv_norm(kv)
                scores = (torch.einsum("bshc,btc->bsht", q_nope, kv_norm) +
                          torch.einsum("bshr,btr->bsht", q_pe, k_pe.squeeze(2))) * self.softmax_scale
        else:
            # Inference mode: use KV cache
            if attn_impl == "naive":
                q = torch.cat([q_nope, q_pe], dim=-1)
                kv = self.wkv_b(self.kv_norm(kv))
                kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
                self.k_cache[:bsz, start_pos:end_pos] = k
                self.v_cache[:bsz, start_pos:end_pos] = v
                scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
            else:
                wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
                wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
                q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
                self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
                scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                          torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # Check if we're in training mode for value computation
        if self.training and seqlen > 1:
            # Training mode: compute without using cache
            if attn_impl == "naive":
                x = torch.einsum("bsht,bthd->bshd", scores, v)
            else:
                x = torch.einsum("bsht,btc->bshc", scores, kv_norm)
                x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        else:
            # Inference mode: use cache
            if attn_impl == "naive":
                x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
            else:
                x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
                x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.n_routed_experts = args.n_routed_experts
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.bfloat16)) if self.dim == 7168 else None
        
        # Add bias-based load balancing components
        self.expert_biases = nn.Parameter(torch.zeros(args.n_routed_experts, dtype=torch.bfloat16))
        self.register_buffer("expert_loads", torch.zeros(args.n_routed_experts))
        self.register_buffer("expert_counts", torch.zeros(args.n_routed_experts))
        self.bias_update_speed = args.bias_update_speed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        
        # Store original scores for gating values (before any bias)
        original_scores = scores.clone()
        
        # Apply structural bias if it exists (legacy behavior for dim=7168)
        if self.bias is not None:
            scores = scores + self.bias
            
        # Create routing scores by adding expert biases ONLY for routing
        routing_scores = scores + self.expert_biases.unsqueeze(0)
        
        if self.n_groups > 1:
            routing_scores = routing_scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = routing_scores.amax(dim=-1)
            else:
                group_scores = routing_scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = routing_scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            routing_scores = routing_scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        # Select top-k experts based on biased routing scores
        indices = torch.topk(routing_scores, self.topk, dim=-1)[1]
        
        # But use original scores for the actual gating weights
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        
        # Update load statistics during training
        if self.training:
            self.update_expert_loads(indices, x.size(0))
        
        return weights.type_as(x), indices
    
    def update_expert_loads(self, indices: torch.Tensor, batch_size: int):
        """Update expert load statistics for bias adjustment"""
        if self.training:
            # Count expert usage in current batch
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
            self.expert_counts += counts.float()
            
            # Update running average of loads
            batch_load = counts.float() / (batch_size * indices.size(1))
            # Initialize expert_loads if None or handle first update
            if self.expert_loads is None:
                self.expert_loads = batch_load
            else:
                self.expert_loads = 0.9 * self.expert_loads + 0.1 * batch_load
            
    def adjust_biases(self):
        """Adjust biases based on expert loads (call at end of training step)"""
        if self.training:
            target_load = 1.0 / self.n_routed_experts
            load_diff = self.expert_loads - target_load
            
            # Decrease bias for overloaded experts, increase for underloaded
            # Apply gradient-like update with momentum for stability
            bias_update = self.bias_update_speed * load_diff
            self.expert_biases.data -= bias_update
            
            # Clamp biases to prevent numerical overflow in bfloat16
            self.expert_biases.data.clamp_(-1e4, 1e4)
            
            # Reset counts periodically to adapt to changing data distribution
            if self.expert_counts.sum() > 10000:
                self.expert_counts.zero_()
                # Optionally decay biases slightly to allow re-adaptation
                self.expert_biases.data *= 0.95


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        
        # Single-pass expert processing with fused operations
        num_local_experts = self.experts_end_idx - self.experts_start_idx
        
        # Process experts in a single loop - build mask and process immediately
        for local_idx in range(num_local_experts):
            global_idx = self.experts_start_idx + local_idx
            
            # Skip if no tokens for this expert
            if counts[global_idx] == 0:
                continue
                
            # Build mask for this expert
            expert_mask = (indices == global_idx)  # [num_tokens, n_activated_experts]
            
            # Find which tokens are routed to this expert using sum (more efficient than any)
            token_indices = expert_mask.sum(dim=1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) > 0:
                expert = self.experts[global_idx]
                
                # Process tokens through expert
                expert_input = x[token_indices]
                expert_output = expert(expert_input)
                
                # Gather weights for these tokens
                # expert_mask[token_indices] gives us which positions in topk selected this expert
                expert_weights = weights[token_indices][expert_mask[token_indices]].unsqueeze(-1)
                
                # Accumulate weighted expert outputs
                y[token_indices] += expert_output * expert_weights
                
        z = self.shared_experts(x)
        
        # Load balancing is handled through expert biases in Gate
        
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MTPModule(nn.Module):
    """
    Multi-Token Prediction module for depth k.
    
    Implements the sequential MTP architecture from DeepSeek-V3 paper:
    - Combines previous hidden state with next token embedding
    - Applies transformer block for prediction at depth k
    """
    def __init__(self, args: ModelArgs, depth: int):
        super().__init__()
        self.depth = depth
        self.dim = args.dim
        
        # Projection to combine previous hidden state with next token embedding
        # M_k in Eq. 21: h'_i^k = M_k * [RMSNorm(h_i^(k-1)); RMSNorm(Emb(t_(i+k)))]
        self.projection = Linear(2 * args.dim, args.dim)
        
        # Transformer block for this depth (TRM_k in Eq. 22)
        # Use negative layer_id to distinguish from main model blocks
        self.block = Block(layer_id=-depth, args=args)
        
        # Normalization layers
        self.norm_h = RMSNorm(args.dim)
        self.norm_emb = RMSNorm(args.dim)
        
    def forward(self, h_prev: torch.Tensor, next_emb: torch.Tensor, 
                start_pos: int, freqs_cis: torch.Tensor, 
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for MTP module at depth k.
        
        Args:
            h_prev: Hidden state from previous depth [batch, seq_len, dim]
            next_emb: Embedding of next token [batch, seq_len, dim]  
            start_pos: Starting position for caching
            freqs_cis: Precomputed RoPE frequencies
            mask: Attention mask
        
        Returns:
            Hidden state for current depth [batch, seq_len, dim]
        """
        # Normalize inputs (Eq. 21)
        h_norm = self.norm_h(h_prev)
        emb_norm = self.norm_emb(next_emb)
        
        # Combine via projection
        combined = torch.cat([h_norm, emb_norm], dim=-1)
        h_combined = self.projection(combined)
        
        # Apply transformer block (Eq. 22)
        h_out = self.block(h_combined, start_pos, freqs_cis, mask)
        
        return h_out


class MTPHead(nn.Module):
    """
    Complete Multi-Token Prediction head.
    
    Implements sequential MTP with shared embedding/output layers.
    For DeepSeek-V3, D=1 (predicts 2 tokens total including main prediction).
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mtp_depth = args.mtp_depth
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        
        # MTP modules for each depth
        self.mtp_modules = nn.ModuleList([
            MTPModule(args, depth=d+1) for d in range(self.mtp_depth)
        ])
        
        # Shared components with main model (will be linked in Transformer)
        self.embed = None  # Will be set to main model's embedding
        self.output_head = None  # Will be set to main model's output head
        
    def forward(self, h_main: torch.Tensor, tokens: torch.Tensor,
                start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor]) -> list:
        """
        Generate multi-token predictions.
        
        Args:
            h_main: Hidden states from main model [batch, seq_len, dim]
            tokens: Input tokens [batch, seq_len]
            start_pos: Starting position
            freqs_cis: RoPE frequencies
            mask: Attention mask
            
        Returns:
            List of predictions for each depth
        """
        batch_size, seq_len = tokens.shape
        predictions = []
        
        h_prev = h_main
        for depth, mtp_module in enumerate(self.mtp_modules):
            # Get embeddings for next tokens (offset by depth+1)
            if depth + 1 < seq_len:
                next_tokens = tokens[:, depth+1:]
                next_emb = self.embed(next_tokens)
                
                # Truncate h_prev to match sequence length
                h_prev_truncated = h_prev[:, :seq_len-depth-1]
                
                # Truncate freqs_cis and mask to match
                truncated_len = seq_len - depth - 1
                freqs_cis_truncated = freqs_cis[:truncated_len] if freqs_cis is not None else None
                mask_truncated = mask[:truncated_len, :truncated_len] if mask is not None else None
                
                # Apply MTP module (Eq. 21-22)
                h_curr = mtp_module(h_prev_truncated, next_emb, 
                                   start_pos, freqs_cis_truncated, mask_truncated)
                
                # Generate predictions (Eq. 23)
                logits = self.output_head(h_curr)
                predictions.append(logits)
                
                # Update h_prev for next depth
                h_prev = h_curr
            
        return predictions


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        # Only set world_size/rank if they haven't been set already
        # This allows tensor parallel code to set them before model creation
        # For DDP, these remain at their default values (1, 0)
        if not hasattr(self, '_world_size_set'):
            # Check if we're being initialized by TensorParallelDeepSeekV3Model
            # If not, we're in DDP mode and should keep world_size=1
            pass  # Keep the default world_size=1, rank=0 for DDP
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self._is_tensor_parallel = False  # Default to False, overridden in TP subclass
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        
        # Weight tying: share embedding weights with output projection
        # Both ParallelEmbedding and ColumnParallelLinear split along vocab dimension
        # so we can directly share the transposed weight
        self.head.weight = self.embed.weight
        
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        
        # Add MTP head if enabled
        self.mtp_enabled = args.mtp_depth > 0
        if self.mtp_enabled:
            self.mtp_head = MTPHead(args)
            # Share embedding and output head with main model
            self.mtp_head.embed = self.embed
            self.mtp_head.output_head = self.head
            
        # MTP loss weight
        self.mtp_lambda = args.mtp_lambda
        
        # Gradient checkpointing flags
        self.gradient_checkpointing = args.gradient_checkpointing
        self.gradient_checkpointing_use_reentrant = args.gradient_checkpointing_use_reentrant

    def gradient_checkpointing_enable(self, use_reentrant: bool = False):
        """
        Enable gradient checkpointing for memory optimization during training.
        
        Args:
            use_reentrant (bool): Whether to use reentrant mode for gradient checkpointing.
                                  Set to False for PyTorch 2.0+ (recommended).
        """
        self.gradient_checkpointing = True
        self.gradient_checkpointing_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing.
        """
        self.gradient_checkpointing = False

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, return_mtp: bool = False, 
                attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.
            return_mtp (bool, optional): Whether to return MTP predictions. Defaults to False.
            attention_mask (Optional[torch.Tensor]): Attention mask for padding tokens. 
                Shape (batch_size, seq_len). 1 for real tokens, 0 for padding.

        Returns:
            If return_mtp is False:
                torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
            If return_mtp is True:
                Tuple[torch.Tensor, list]: Main logits and list of MTP predictions.
        """
        batch_size = tokens.size(0)
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        
        # Handle padding by zeroing out embeddings for padding tokens
        if attention_mask is not None:
            # Zero out embeddings for padding tokens
            # This is a simple but effective approach
            # Ensure attention_mask has the same dtype as h
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)
        
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            # Create causal mask - this handles the autoregressive property
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        
        # Apply transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                h = checkpoint(layer, h, start_pos, freqs_cis, mask,
                             use_reentrant=self.gradient_checkpointing_use_reentrant)
            else:
                # Normal forward pass without checkpointing
                h = layer(h, start_pos, freqs_cis, mask)
        
        # Store hidden states before normalization for MTP
        # Clone to avoid sharing computation graph
        h_for_mtp = h.clone() if self.mtp_enabled and return_mtp and self.training else None
        
        # Get main prediction
        # For training, compute logits for all positions; for inference, just the last
        if self.training:
            h_final = self.norm(h)  # [batch_size, seq_len, dim]
            logits_main = self.head(h_final)  # [batch_size, seq_len, vocab_size]
        else:
            h_final = self.norm(h)[:, -1]  # [batch_size, dim]
            logits_main = self.head(h_final)  # [batch_size, vocab_size]
        
        # Get MTP predictions if enabled and requested
        mtp_logits = []
        if self.mtp_enabled and return_mtp and self.training and seqlen > 1 and h_for_mtp is not None:
            # Apply final norm to all positions for MTP
            h_normed = self.norm(h_for_mtp)
            mtp_logits = self.mtp_head(h_normed, tokens, start_pos, freqs_cis, mask)
        
        # Gather logits across devices if distributed
        # IMPORTANT: Only do this for tensor parallelism, not DDP!
        # In TP, vocabulary is split across devices and needs gathering
        # In DDP, each device has the full model and full vocabulary
        if world_size > 1 and hasattr(self, '_is_tensor_parallel') and self._is_tensor_parallel:
            all_logits = [torch.empty_like(logits_main) for _ in range(world_size)]
            dist.all_gather(all_logits, logits_main)
            logits_main = torch.cat(all_logits, dim=-1)
            
            # Also gather MTP logits
            if mtp_logits:
                for i, mtp_pred in enumerate(mtp_logits):
                    all_mtp = [torch.empty_like(mtp_pred) for _ in range(world_size)]
                    dist.all_gather(all_mtp, mtp_pred)
                    mtp_logits[i] = torch.cat(all_mtp, dim=-1)
        
        if return_mtp:
            return logits_main, mtp_logits
        return logits_main
    
    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Inference-only forward pass (original method for generation).
        
        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        # Call forward without MTP for inference
        return self.forward(tokens, start_pos, return_mtp=False)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
