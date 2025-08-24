#!/usr/bin/env python3
"""
Calculate optimal training steps based on dataset size and training parameters.
"""

import json
import argparse
import os
import re
from pathlib import Path

# Dataset token counts (hardcoded from dataset_tokens_fast.log)
DATASET_TOKENS = {
    "books1": 1_543_681_566,
    "books3": 25_600_991_416,
    "europarl": 1_373_177_908,
    "github": 1_654_916_041,
    "institutional_books": 241_728_066_485,
    "pmc_extracts": 22_174_580_797,
    "literotica": 2_872_750_976,
    "nih_exporter": 380_872_466,
    "man_info_pages": 64_706_818,
}

def parse_dataset_log(log_file):
    """Parse dataset_tokens_fast.log file for token counts."""
    if not os.path.exists(log_file):
        return {}
    
    dataset_info = {}
    current_dataset = None
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Dataset name
            if line.endswith(':') and not line.startswith(' '):
                current_dataset = line[:-1]
                dataset_info[current_dataset] = {}
            
            # Token count
            elif 'Tokens:' in line or 'Total tokens:' in line:
                tokens_str = line.split(':')[1].strip()
                tokens = int(tokens_str.replace(',', ''))
                if current_dataset:
                    dataset_info[current_dataset]['tokens'] = tokens
    
    return dataset_info

def get_dataset_tokens(data_dir):
    """Get token count for a dataset directory."""
    # Extract dataset name from path
    data_dir = Path(data_dir)
    dataset_name = data_dir.name
    
    # Check if we have hardcoded token count
    if dataset_name in DATASET_TOKENS:
        return DATASET_TOKENS[dataset_name]
    
    # Try to parse from log file
    log_file = "archive/logs/dataset_tokens_fast.log"
    if os.path.exists(log_file):
        dataset_info = parse_dataset_log(log_file)
        if dataset_name in dataset_info and 'tokens' in dataset_info[dataset_name]:
            return dataset_info[dataset_name]['tokens']
    
    # If not found, estimate based on file size (rough approximation)
    total_size = 0
    for file in data_dir.glob("**/*"):
        if file.is_file():
            total_size += file.stat().st_size
    
    # Rough estimate: 1 token ≈ 4 bytes (varies by tokenizer)
    estimated_tokens = total_size // 4
    return estimated_tokens

def calculate_steps(tokens, seq_len, batch_size, accumulation_steps, num_epochs=1):
    """Calculate number of training steps."""
    # Effective batch size
    effective_batch_size = batch_size * accumulation_steps
    
    # Tokens per step
    tokens_per_step = seq_len * effective_batch_size
    
    # Total steps
    steps_per_epoch = int(tokens // tokens_per_step)
    total_steps = int(steps_per_epoch * num_epochs)
    
    return {
        'tokens': tokens,
        'tokens_per_step': tokens_per_step,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'num_epochs': num_epochs,
        'effective_batch_size': effective_batch_size
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate training steps based on dataset size')
    parser.add_argument('--data-dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--seq-len', type=int, default=4096, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--num-epochs', type=float, default=1, help='Number of epochs')
    parser.add_argument('--warmup-ratio', type=float, default=0.05, help='Warmup ratio (default: 5%)')
    parser.add_argument('--output-json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    # Get token count
    tokens = get_dataset_tokens(args.data_dir)
    
    # Calculate steps
    info = calculate_steps(
        tokens=tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        num_epochs=args.num_epochs
    )
    
    # Calculate warmup steps
    warmup_steps = int(info['total_steps'] * args.warmup_ratio)
    info['warmup_steps'] = warmup_steps
    
    if args.output_json:
        print(json.dumps(info, indent=2))
    else:
        print(f"Dataset: {args.data_dir}")
        print(f"Total tokens: {info['tokens']:,}")
        print(f"Sequence length: {args.seq_len}")
        print(f"Batch size: {args.batch_size}")
        print(f"Accumulation steps: {args.accumulation_steps}")
        print(f"Effective batch size: {info['effective_batch_size']}")
        print(f"Tokens per step: {info['tokens_per_step']:,}")
        print(f"Steps per epoch: {info['steps_per_epoch']:,}")
        print(f"Number of epochs: {info['num_epochs']}")
        print(f"Total steps: {info['total_steps']:,}")
        print(f"Warmup steps ({args.warmup_ratio*100:.0f}%): {info['warmup_steps']:,}")

if __name__ == "__main__":
    main()
