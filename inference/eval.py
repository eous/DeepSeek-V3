#!/usr/bin/env python3
"""
Evaluation script for DeepSeek-V3 model checkpoints.

This script loads a trained checkpoint and generates text from a given prompt,
displaying the main tokens in green and MTP predictions in red.
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os

# Import model components
from inference.model import Transformer, ModelArgs

# Import rich for colored output
try:
    from rich.console import Console
    from rich.text import Text
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not available. Install with 'pip install rich' for colored output.")

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: str = 'cuda'):
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Print checkpoint metadata if available
        if 'global_step' in checkpoint:
            print(f"Checkpoint from step: {checkpoint['global_step']}")
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'args' in checkpoint:
            args = checkpoint['args']
            if 'learning_rate' in args:
                print(f"Training LR: {args['learning_rate']}")
            if 'batch_size' in args:
                print(f"Training batch size: {args['batch_size']}")
    else:
        state_dict = checkpoint
    
    # Handle DDP state dict (module prefix)
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("Detected DDP checkpoint, removing 'module.' prefix...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value  # Remove 'module.' prefix
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Try to load with strict=False to handle mismatches
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully!")
    except RuntimeError as e:
        print(f"Warning: Loading checkpoint with strict=False due to mismatch: {str(e)[:200]}...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} keys (likely MTP components)")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)} keys")
        print("Checkpoint loaded with warnings. Model may not perform as expected.")
    
    return model

def create_model_from_config(config_path: str, checkpoint_path: str = None, device: str = 'cuda'):
    """Create model from configuration file or checkpoint."""
    config = None
    
    # First try to get config from checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'config' in checkpoint:
                print(f"Using configuration from checkpoint...")
                config = checkpoint['config']
        except Exception as e:
            print(f"Could not load config from checkpoint: {e}")
    
    # Fall back to config file
    if config is None:
        print(f"Loading model configuration from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Convert config dict to ModelArgs
    model_args = ModelArgs(
        vocab_size=config.get('vocab_size', 128256),
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
        max_batch_size=config.get('max_batch_size', 8)
    )
    
    # Create model
    model = Transformer(model_args)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model, model_args

class SamplingConfig:
    """Configuration for advanced sampling techniques."""
    def __init__(self, 
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 min_p: float = 0.0,
                 typical_p: float = 1.0,
                 eta_cutoff: float = 0.0,
                 epsilon_cutoff: float = 0.0,
                 repetition_penalty: float = 1.0,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0,
                 dry_multiplier: float = 0.0,
                 dry_allowed_length: int = 2,
                 dry_base: float = 1.75,
                 dry_sequence_breakers: list = None,
                 xtc_threshold: float = 0.1,
                 xtc_probability: float = 0.5):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.typical_p = typical_p
        self.eta_cutoff = eta_cutoff
        self.epsilon_cutoff = epsilon_cutoff
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.dry_multiplier = dry_multiplier
        self.dry_allowed_length = dry_allowed_length
        self.dry_base = dry_base
        self.dry_sequence_breakers = dry_sequence_breakers or ["\n", ".", "!", "?", ","]
        self.xtc_threshold = xtc_threshold
        self.xtc_probability = xtc_probability
    
    @classmethod
    def preset_conservative(cls):
        """Conservative preset - high quality, low diversity."""
        return cls(
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.1
        )
    
    @classmethod
    def preset_creative(cls):
        """Creative preset - more diverse outputs."""
        return cls(
            temperature=1.2,
            top_p=0.95,
            typical_p=0.95,
            repetition_penalty=1.05,
            dry_multiplier=0.8,
            dry_allowed_length=3
        )
    
    @classmethod
    def preset_precise(cls):
        """Precise preset - deterministic, high quality."""
        return cls(
            temperature=0.3,
            top_k=10,
            top_p=0.85,
            min_p=0.1,
            repetition_penalty=1.15
        )
    
    @classmethod
    def preset_balanced(cls):
        """Balanced preset - good for general use."""
        return cls(
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            min_p=0.02,
            typical_p=0.98,
            repetition_penalty=1.05,
            dry_multiplier=0.5,
            dry_allowed_length=2
        )
    
    @classmethod
    def preset_experimental(cls):
        """Experimental preset - uses advanced techniques."""
        return cls(
            temperature=0.9,
            top_p=0.9,
            min_p=0.03,
            typical_p=0.95,
            eta_cutoff=0.0003,
            epsilon_cutoff=0.0003,
            repetition_penalty=1.1,
            dry_multiplier=1.0,
            dry_allowed_length=2,
            dry_base=1.75,
            xtc_threshold=0.1,
            xtc_probability=0.5
        )

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    if temperature <= 0:
        return logits
    return logits / temperature

def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits."""
    if k <= 0:
        return logits
    
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    min_value = values[..., -1].unsqueeze(-1)
    return torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)

def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits."""
    if p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float('-inf'))

def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Apply min-p filtering to logits."""
    if min_p <= 0.0:
        return logits
    
    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    scaled_min_p = min_p * max_prob
    return torch.where(probs < scaled_min_p, torch.full_like(logits, float('-inf')), logits)

def apply_typical_p(logits: torch.Tensor, p: float, mass: float = 0.9) -> torch.Tensor:
    """Apply typical sampling (locally typical sampling)."""
    if p >= 1.0:
        return logits
    
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
    
    # Calculate negative log likelihood
    neg_log_probs = -torch.log(probs + 1e-10)
    
    # Calculate absolute difference from entropy (typicality)
    diff = torch.abs(neg_log_probs - entropy)
    
    # Sort by typicality (ascending)
    sorted_diff, sorted_indices = torch.sort(diff)
    sorted_probs = probs.gather(-1, sorted_indices)
    
    # Find cutoff where cumulative probability exceeds mass
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_index = (cumsum_probs < mass).sum(dim=-1, keepdim=True)
    
    # Create mask for tokens to keep
    indices = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0)
    mask = indices <= cutoff_index
    
    # Apply mask to original logits
    keep_indices = sorted_indices.masked_select(mask)
    new_logits = torch.full_like(logits, float('-inf'))
    new_logits.scatter_(-1, keep_indices, logits.gather(-1, keep_indices))
    
    return new_logits

def apply_eta_cutoff(logits: torch.Tensor, eta: float) -> torch.Tensor:
    """Apply eta cutoff (entropy-based truncation)."""
    if eta <= 0.0:
        return logits
    
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    # Calculate dynamic cutoff based on entropy
    max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float))
    cutoff = eta * entropy / max_entropy
    
    # Apply cutoff
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find where cumulative probability exceeds cutoff
    indices_to_remove = cumulative_probs > cutoff.unsqueeze(-1)
    indices_to_remove[..., 0] = False  # Keep at least one token
    
    # Scatter back to original indices
    indices_to_remove = indices_to_remove.scatter(-1, sorted_indices, indices_to_remove)
    return logits.masked_fill(indices_to_remove, float('-inf'))

def apply_epsilon_cutoff(logits: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Apply epsilon cutoff (probability-based cutoff)."""
    if epsilon <= 0.0:
        return logits
    
    probs = torch.softmax(logits, dim=-1)
    indices_to_remove = probs < epsilon
    
    # Keep at least one token (the highest probability one)
    max_prob_idx = probs.argmax(dim=-1)
    indices_to_remove.scatter_(-1, max_prob_idx.unsqueeze(-1), False)
    
    return logits.masked_fill(indices_to_remove, float('-inf'))

def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """Apply repetition penalty to discourage repeating tokens."""
    if penalty == 1.0:
        return logits
    
    # Get unique tokens in the input
    unique_ids = input_ids.unique()
    
    # Apply penalty
    for token_id in unique_ids:
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty
    
    return logits

def apply_dry_penalty(logits: torch.Tensor, input_ids: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Apply DRY (Don't Repeat Yourself) penalty for longer sequence repetitions."""
    if config.dry_multiplier <= 0.0:
        return logits
    
    seq_len = len(input_ids)
    if seq_len < config.dry_allowed_length:
        return logits
    
    # Find repeated sequences
    for length in range(config.dry_allowed_length, min(seq_len // 2, 50)):
        for i in range(seq_len - length * 2 + 1):
            seq1 = input_ids[i:i + length]
            seq2 = input_ids[i + length:i + length * 2]
            
            if torch.equal(seq1, seq2):
                # Apply penalty to tokens that would continue the repetition
                if i + length * 2 < seq_len:
                    next_token = input_ids[i + length * 2]
                    penalty = config.dry_base ** (length - config.dry_allowed_length) * config.dry_multiplier
                    if logits[next_token] < 0:
                        logits[next_token] *= (1 + penalty)
                    else:
                        logits[next_token] /= (1 + penalty)
    
    return logits

def apply_xtc(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Apply XTC (eXclude Top Choices) sampling."""
    if config.xtc_probability <= 0.0 or config.xtc_threshold <= 0.0:
        return logits
    
    # Random chance to apply XTC
    if torch.rand(1).item() > config.xtc_probability:
        return logits
    
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Find tokens above threshold
    above_threshold = sorted_probs > config.xtc_threshold
    if above_threshold.sum() <= 1:
        return logits  # Keep at least one token
    
    # Exclude top tokens above threshold (except the very top one)
    n_exclude = above_threshold.sum().item() - 1
    if n_exclude > 0:
        exclude_indices = sorted_indices[:n_exclude]
        logits[exclude_indices] = float('-inf')
    
    return logits

def sample_from_logits(logits: torch.Tensor, config: SamplingConfig, 
                      input_ids: torch.Tensor = None) -> torch.Tensor:
    """Apply all sampling techniques and sample a token."""
    # Apply repetition penalties if input_ids provided
    if input_ids is not None and len(input_ids) > 0:
        logits = apply_repetition_penalty(logits, input_ids, config.repetition_penalty)
        logits = apply_dry_penalty(logits, input_ids, config)
    
    # Apply temperature
    logits = apply_temperature(logits, config.temperature)
    
    # Apply filtering techniques (order matters!)
    logits = apply_epsilon_cutoff(logits, config.epsilon_cutoff)
    logits = apply_eta_cutoff(logits, config.eta_cutoff)
    logits = apply_typical_p(logits, config.typical_p)
    logits = apply_xtc(logits, config)
    logits = apply_top_k(logits, config.top_k)
    logits = apply_min_p(logits, config.min_p)
    logits = apply_top_p(logits, config.top_p)
    
    # Convert to probabilities and sample
    probs = torch.softmax(logits, dim=-1)
    
    # Handle the case where all probabilities are 0 (shouldn't happen with proper filtering)
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.size(-1)
    
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate(model, tokenizer, prompt: str, num_tokens: int, config: SamplingConfig = None, seed: int = 42):
    """Generate text from prompt, returning main and MTP predictions."""
    # Use default config if none provided
    if config is None:
        config = SamplingConfig()
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = next(model.parameters()).device
    model.train()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = input_ids.unsqueeze(0)
    
    print(f"\nGenerating {num_tokens} tokens from prompt: '{prompt}'")
    print(f"Sampling config: temperature={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")
    print(f"Advanced: min_p={config.min_p}, typical_p={config.typical_p}, repetition_penalty={config.repetition_penalty}")
    print(f"Seed: {seed}")
    print("-" * 80)
    
    generated_tokens = []
    mtp_predictions = []
    
    # Check if model has MTP enabled
    has_mtp = hasattr(model, 'mtp_enabled') and model.mtp_enabled and hasattr(model, 'mtp_head') and model.mtp_head is not None
    
    for i in range(num_tokens):
        # Get model predictions
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            if has_mtp:
                logits, mtp_logits = model(tokens, start_pos=0, return_mtp=True)
            else:
                logits = model(tokens, start_pos=0)
                mtp_logits = None
        
        # Get the last token's logits
        if logits.dim() == 3:
            next_token_logits = logits[0, -1, :]
        else:
            next_token_logits = logits[0, :]
        
        # Apply advanced sampling
        next_token = sample_from_logits(next_token_logits, config, input_ids)
        
        # Extract the actual token value
        if next_token.dim() > 0:
            next_token_value = next_token.item()
        else:
            next_token_value = next_token
            
        generated_tokens.append(next_token_value)
        
        # Handle MTP predictions if available
        if mtp_logits and len(mtp_logits) > 0:
            # Get predictions for each MTP head
            mtp_tokens_for_position = []
            for head_idx, head_logits in enumerate(mtp_logits):
                if head_logits.dim() == 3:
                    # Take the last sequence position
                    head_logits = head_logits[0, -1, :]
                else:
                    head_logits = head_logits[0, :]
                
                # Apply same sampling config to MTP predictions
                mtp_token = sample_from_logits(head_logits, config, input_ids)
                mtp_tokens_for_position.append(mtp_token.item())
            
            mtp_predictions.append(mtp_tokens_for_position)
        else:
            mtp_predictions.append([])
        
        # Update tokens for next iteration
        next_token_tensor = torch.tensor([[next_token_value]], dtype=torch.long, device=device)
        tokens = torch.cat([tokens, next_token_tensor], dim=1)
        input_ids = torch.cat([input_ids, torch.tensor([next_token_value], device=device)])
        
        # Decode and display the generated token
        main_token_str = tokenizer.decode([next_token_value])
        
        if RICH_AVAILABLE:
            # Create colored text
            text = Text()
            text.append(f"Token {i+1}: ", style="dim")
            text.append(main_token_str, style="green bold")
            
            # Add MTP predictions for NEXT tokens if available
            if mtp_predictions[-1]:
                text.append(" → Next predictions: ", style="dim")
                for j, mtp_token in enumerate(mtp_predictions[-1]):
                    if j > 0:
                        text.append(", ", style="dim")
                    mtp_token_str = tokenizer.decode([mtp_token])
                    text.append(f"[{j+1}] {mtp_token_str}", style="red bold")
            
            console.print(text)
        else:
            # Plain text output
            print(f"Token {i+1}: {main_token_str}", end="")
            if mtp_predictions[-1]:
                mtp_strs = [tokenizer.decode([t]) for t in mtp_predictions[-1]]
                print(f" → Next predictions: {', '.join([f'[{j+1}] {s}' for j, s in enumerate(mtp_strs)])}")
            else:
                print()  # New line
        
        sys.stdout.flush()
    
    print("\n" + "-" * 80)
    
    # Decode full generated text
    full_text = tokenizer.decode(generated_tokens)
    
    if RICH_AVAILABLE:
        console.print("\n[bold]Complete generated sequence:[/bold]")
        console.print(f"[cyan]{prompt}[/cyan][green]{full_text}[/green]")
    else:
        print(f"\nComplete generated sequence:")
        print(f"{prompt}{full_text}")
    
    return generated_tokens, mtp_predictions

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoint files
    checkpoints = list(checkpoint_dir.glob("checkpoint-step-*.pt"))
    if not checkpoints:
        # Try .pth extension
        checkpoints = list(checkpoint_dir.glob("checkpoint-step-*.pth"))
    
    if not checkpoints:
        return None
    
    # Sort by step number
    def get_step(path):
        try:
            return int(path.stem.split('-')[-1])
        except:
            return -1
    
    checkpoints.sort(key=get_step, reverse=True)
    return str(checkpoints[0])

def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepSeek-V3 model checkpoint with advanced sampling')
    
    # Model and checkpoint arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration JSON file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint file or directory (will use latest if directory)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory to search for latest checkpoint (default: checkpoints)')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str, required=True,
                        help='Input prompt for generation')
    parser.add_argument('--num_tokens', type=int, required=True,
                        help='Number of tokens to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (default: cuda)')
    
    # Preset configuration
    parser.add_argument('--preset', type=str, default=None,
                        choices=['conservative', 'creative', 'precise', 'balanced', 'experimental'],
                        help='Use a preset sampling configuration')
    
    # Basic sampling arguments
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k filtering (0 = disabled, default: 0)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p (nucleus) sampling (default: 1.0)')
    parser.add_argument('--min_p', type=float, default=0.0,
                        help='Min-p filtering (default: 0.0)')
    
    # Advanced sampling arguments
    parser.add_argument('--typical_p', type=float, default=1.0,
                        help='Typical sampling threshold (default: 1.0)')
    parser.add_argument('--eta_cutoff', type=float, default=0.0,
                        help='Eta cutoff for entropy-based truncation (default: 0.0)')
    parser.add_argument('--epsilon_cutoff', type=float, default=0.0,
                        help='Epsilon cutoff for probability-based truncation (default: 0.0)')
    
    # Repetition penalty arguments
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty (default: 1.0)')
    parser.add_argument('--presence_penalty', type=float, default=0.0,
                        help='Presence penalty (default: 0.0)')
    parser.add_argument('--frequency_penalty', type=float, default=0.0,
                        help='Frequency penalty (default: 0.0)')
    
    # DRY (Don't Repeat Yourself) penalty arguments
    parser.add_argument('--dry_multiplier', type=float, default=0.0,
                        help='DRY penalty multiplier (default: 0.0)')
    parser.add_argument('--dry_allowed_length', type=int, default=2,
                        help='Minimum sequence length before applying DRY penalty (default: 2)')
    parser.add_argument('--dry_base', type=float, default=1.75,
                        help='DRY penalty base for exponential scaling (default: 1.75)')
    
    # XTC (eXclude Top Choices) arguments
    parser.add_argument('--xtc_threshold', type=float, default=0.1,
                        help='XTC threshold for excluding top choices (default: 0.1)')
    parser.add_argument('--xtc_probability', type=float, default=0.0,
                        help='Probability of applying XTC (0.0 = disabled, default: 0.0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Handle checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try to find latest checkpoint in checkpoint_dir
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            print(f"Error: No checkpoints found in {args.checkpoint_dir}")
            sys.exit(1)
        print(f"Using latest checkpoint: {checkpoint_path}")
    elif Path(checkpoint_path).is_dir():
        # If a directory is provided, find latest checkpoint in it
        latest = find_latest_checkpoint(checkpoint_path)
        if latest is None:
            print(f"Error: No checkpoints found in directory: {checkpoint_path}")
            sys.exit(1)
        checkpoint_path = latest
        print(f"Using latest checkpoint from directory: {checkpoint_path}")
    elif not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    if args.num_tokens <= 0:
        print("Error: num_tokens must be positive")
        sys.exit(1)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3",
            trust_remote_code=True
        )
        # Fix pad token
        tokenizer.pad_token_id = 2
        print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure you have transformers installed and internet connection")
        sys.exit(1)
    
    # Create and load model
    model, model_args = create_model_from_config(args.config, checkpoint_path, args.device)
    model = load_checkpoint(checkpoint_path, model, args.device)
    
    # Create sampling configuration
    if args.preset:
        # Use preset configuration
        preset_map = {
            'conservative': SamplingConfig.preset_conservative,
            'creative': SamplingConfig.preset_creative,
            'precise': SamplingConfig.preset_precise,
            'balanced': SamplingConfig.preset_balanced,
            'experimental': SamplingConfig.preset_experimental
        }
        sampling_config = preset_map[args.preset]()
        print(f"\nUsing preset: {args.preset}")
    else:
        # Use custom configuration
        sampling_config = SamplingConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            typical_p=args.typical_p,
            eta_cutoff=args.eta_cutoff,
            epsilon_cutoff=args.epsilon_cutoff,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            dry_multiplier=args.dry_multiplier,
            dry_allowed_length=args.dry_allowed_length,
            dry_base=args.dry_base,
            xtc_threshold=args.xtc_threshold,
            xtc_probability=args.xtc_probability
        )
    
    # Generate text
    generated_tokens, mtp_predictions = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        config=sampling_config,
        seed=args.seed
    )
    
    print("\nGeneration complete!")

if __name__ == "__main__":
    main()
