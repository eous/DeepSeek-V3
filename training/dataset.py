"""
Iterable dataset for training DeepSeek-V3 on parquet files.

Supports:
- Multiple data directories
- Efficient tokenization with caching
- Proper attention masks
- Worker-based parallelism
- Memory-efficient streaming
"""

import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
import pandas as pd
from typing import Union, List
from pathlib import Path
from typing import List, Dict, Optional, Iterator
import logging
from collections import OrderedDict
import hashlib


class ParquetTextDataset(IterableDataset):
    """
    Iterable dataset for streaming text data from parquet files.
    
    Features:
    - Handles multiple dataset directories
    - Efficient tokenization with LRU cache
    - Proper sequence packing with attention masks
    - Multi-worker support for parallel loading
    - Memory-efficient streaming
    """
    
    def __init__(
        self,
        tokenizer,
        data_dirs: Union[str, List[str]],
        seq_len: int = 2048,
        logger: Optional[logging.Logger] = None,
        cache_size: int = 1000,
        buffer_size: int = 10000,
        add_eos_token: bool = True,
        stride: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: HuggingFace tokenizer
            data_dirs: Comma-separated list of data directories
            seq_len: Sequence length for training
            logger: Logger instance
            cache_size: Size of tokenization cache
            buffer_size: Size of sample buffer
            add_eos_token: Whether to add EOS token at document boundaries
            stride: Stride for sliding window (None = no overlap)
            rank: Process rank for distributed training
            world_size: Total number of processes for distributed training
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.logger = logger or logging.getLogger(__name__)
        self.cache_size = cache_size
        self.buffer_size = buffer_size
        self.add_eos_token = add_eos_token
        self.stride = stride or seq_len  # Default: no overlap
        self.rank = rank
        self.world_size = world_size
        
        # CRITICAL: Fix pad token if needed
        # DeepSeek-V3 tokenizer defaults to pad_token_id=1 (same as EOS)
        # But there's a dedicated PAD token at ID 2
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.logger.warning(
                f"Tokenizer has pad_token_id == eos_token_id ({self.tokenizer.pad_token_id}). "
                "This is likely incorrect. Consider setting tokenizer.pad_token_id = 2"
            )
        
        # Token IDs from tokenizer
        self.pad_token_id = tokenizer.pad_token_id  # 2 for DeepSeek-V3
        self.bos_token_id = tokenizer.bos_token_id  # 0 for DeepSeek-V3
        self.eos_token_id = tokenizer.eos_token_id  # 1 for DeepSeek-V3
        
        # Parse data directories
        if isinstance(data_dirs, list):
            self.data_dirs = [Path(d) for d in data_dirs]
        elif ',' in data_dirs:
            self.data_dirs = [Path(d.strip()) for d in data_dirs.split(',')]
        else:
            self.data_dirs = [Path(data_dirs)]
        
        # Find all parquet files
        self.files = []
        self.dataset_info = {}  # Track dataset for each file
        self._scan_directories()
        
        if not self.files:
            raise ValueError(f"No parquet files found in: {data_dirs}")
        
        # Initialize tokenization cache (shared across workers in same process)
        self._token_cache = OrderedDict()
        self._cache_stats = {'hits': 0, 'misses': 0}
    
    def _scan_directories(self):
        """Scan directories for parquet files with multiple structure support."""
        for data_dir in self.data_dirs:
            # Try standardized structure: dataset/data/*.parquet
            if (data_dir / "data").exists():
                self._add_files_from_dir(data_dir / "data", data_dir.name)
            else:
                # Try nested structure: root/*/data/*.parquet
                found_any = False
                for subdir in sorted(data_dir.iterdir()):
                    if subdir.is_dir() and (subdir / "data").exists():
                        self._add_files_from_dir(subdir / "data", subdir.name)
                        found_any = True
                
                # Fallback to flat structure
                if not found_any:
                    self._add_files_from_dir(data_dir, data_dir.name)
        
        # Distribute files across ranks for DDP
        if self.world_size > 1:
            # Sort files to ensure consistent ordering across ranks
            self.files = sorted(self.files)
            # Assign files to ranks in round-robin fashion
            self.files = self.files[self.rank::self.world_size]
            self.logger.info(f"Rank {self.rank}/{self.world_size}: Assigned {len(self.files)} files")
        
        # Log dataset summary
        self._log_dataset_summary()
    
    def _add_files_from_dir(self, directory: Path, dataset_name: str):
        """Add parquet files from a directory."""
        files = sorted(directory.glob("*.parquet"))
        if files:
            self.logger.info(f"Found {len(files)} files in dataset '{dataset_name}'")
            for f in files:
                self.files.append(f)
                self.dataset_info[str(f)] = dataset_name
    
    def _log_dataset_summary(self):
        """Log summary of loaded datasets."""
        dataset_counts = {}
        for dataset in self.dataset_info.values():
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        self.logger.info(f"Total files: {len(self.files)} from {len(dataset_counts)} datasets:")
        for dataset, count in sorted(dataset_counts.items()):
            self.logger.info(f"  - {dataset}: {count} files")
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text with caching and intelligent chunking for long documents."""
        # Create cache key from text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if text_hash in self._token_cache:
            self._cache_stats['hits'] += 1
            # Move to end (LRU)
            self._token_cache.move_to_end(text_hash)
            return self._token_cache[text_hash].copy()
        
        # Tokenize
        self._cache_stats['misses'] += 1
        
        # First attempt normal tokenization
        try:
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,  # Adds BOS automatically
                truncation=False,
                return_tensors=None
            )
            
            # If within reasonable limits, use as-is
            if len(tokens) <= 100000:  # Well under the 131k limit
                # Add to cache
                self._token_cache[text_hash] = tokens.copy()
                
                # Maintain cache size
                if len(self._token_cache) > self.cache_size:
                    # Remove oldest (first) item
                    self._token_cache.popitem(last=False)
                
                return tokens
                
        except Exception as e:
            # Tokenization failed, likely due to extreme length
            if self.logger:
                self.logger.warning(f"Tokenization failed for document with {len(text)} chars: {e}")
        
        # Handle very long documents by chunking at character level
        # This preserves all content without truncation
        if self.logger:
            self.logger.info(f"Chunking long document ({len(text)} chars)")
        
        all_tokens = []
        chunk_size = 400000  # Characters, not tokens - roughly 80-100k tokens
        overlap = 1000  # Character overlap to maintain context
        
        # Add BOS token at the start
        if self.tokenizer.bos_token_id is not None:
            all_tokens.append(self.tokenizer.bos_token_id)
        
        # Process chunks
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            try:
                # Tokenize chunk without special tokens (we'll add them manually)
                chunk_tokens = self.tokenizer.encode(
                    chunk_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=100000,  # Safety limit per chunk
                    return_tensors=None
                )
                
                # For chunks after the first, skip overlapping tokens
                if i > 0 and len(all_tokens) > 0:
                    # Estimate overlap in tokens (rough approximation)
                    overlap_tokens = min(100, len(chunk_tokens) // 10)
                    chunk_tokens = chunk_tokens[overlap_tokens:]
                
                all_tokens.extend(chunk_tokens)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to tokenize chunk at position {i}: {e}")
                continue
        
        # Log result
        if self.logger:
            self.logger.info(f"Chunked document: {len(text)} chars -> {len(all_tokens)} tokens")
        
        # Don't cache very long documents to save memory
        if len(all_tokens) <= 200000:  # Only cache if not too huge
            self._token_cache[text_hash] = all_tokens.copy()
            
            # Maintain cache size
            if len(self._token_cache) > self.cache_size:
                self._token_cache.popitem(last=False)
        
        return all_tokens
    
    def _create_samples(self, tokens: List[int]) -> Iterator[Dict[str, torch.Tensor]]:
        """Create training samples from token list."""
        # Add EOS token if requested (document boundary)
        if self.add_eos_token and len(tokens) > 0 and tokens[-1] != self.eos_token_id:
            tokens.append(self.eos_token_id)
        
        # Handle empty or very short documents
        if len(tokens) <= 1:
            return
        
        # Create overlapping sequences with stride
        num_samples = 0
        for i in range(0, len(tokens) - 1, self.stride):
            # Get chunk (need seq_len + 1 for input/label pairs)
            chunk = tokens[i:i + self.seq_len + 1]
            
            # Skip if chunk is too short (less than 2 tokens)
            if len(chunk) < 2:
                continue
            
            # Pad if necessary
            if len(chunk) < self.seq_len + 1:
                padding_length = self.seq_len + 1 - len(chunk)
                chunk.extend([self.pad_token_id] * padding_length)
            
            # Convert to tensors
            chunk_tensor = torch.tensor(chunk, dtype=torch.long)
            input_ids = chunk_tensor[:-1]
            labels = chunk_tensor[1:]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            # Note: In DeepSeek-V3, PAD token = EOS token, so we need to be careful
            attention_mask = torch.ones_like(input_ids)
            
            # Mark padding positions (after the actual content)
            # Find the last position before padding starts
            if self.pad_token_id in chunk[:-1]:  # Check in input_ids part
                # Find first occurrence of continuous padding at the end
                for j in range(len(input_ids) - 1, -1, -1):
                    if input_ids[j] != self.pad_token_id:
                        # Everything after j+1 is padding
                        if j + 1 < len(input_ids):
                            attention_mask[j + 1:] = 0
                        break
                else:
                    # All tokens are padding (shouldn't happen, but handle it)
                    attention_mask.fill_(0)
            
            yield {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
            
            num_samples += 1
            
            # If we've processed all tokens, break
            if i + self.seq_len >= len(tokens) - 1:
                break
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset."""
        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_per_worker = self.files[worker_id::num_workers]
        else:
            files_per_worker = self.files
        
        # Process files assigned to this worker
        for file_idx, file_path in enumerate(files_per_worker):
            try:
                # Read parquet file
                table = pq.read_table(file_path)
                df = table.to_pandas()
                dataset_name = self.dataset_info[str(file_path)]
                
                # Log progress periodically
                if file_idx % 10 == 0:
                    self.logger.debug(f"Worker processing file {file_idx+1}/{len(files_per_worker)}: {file_path.name}")
                
                # Process each row
                for row_idx, row in df.iterrows():
                    # Get text from various possible column names
                    text = row.get('text', '') or row.get('content', '') or row.get('document', '')
                    if not text or not isinstance(text, str):
                        continue
                    
                    # Skip empty texts but allow short ones for testing
                    if not text:
                        continue
                    
                    # Tokenize text
                    tokens = self._tokenize_text(text)
                    
                    # Skip if too short after tokenization
                    if len(tokens) < self.seq_len // 4:
                        continue
                    
                    # Yield samples from this document
                    for sample in self._create_samples(tokens):
                        yield sample
                
                # Log cache statistics periodically
                if file_idx % 50 == 0 and file_idx > 0:
                    hit_rate = self._cache_stats['hits'] / max(1, self._cache_stats['hits'] + self._cache_stats['misses'])
                    self.logger.debug(f"Tokenization cache hit rate: {hit_rate:.2%}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'num_files': len(self.files),
            'num_datasets': len(set(self.dataset_info.values())),
            'cache_hit_rate': self._cache_stats['hits'] / max(1, self._cache_stats['hits'] + self._cache_stats['misses']),
            'cache_size': len(self._token_cache)
        }


def create_dataloader(
    tokenizer,
    data_dirs: Union[str, List[str]],
    batch_size: int,
    seq_len: int = 2048,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    logger: Optional[logging.Logger] = None,
    rank: int = 0,
    world_size: int = 1
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the ParquetTextDataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        data_dirs: Either a comma-separated string or list of data directories
        batch_size: Batch size
        seq_len: Sequence length
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        logger: Logger instance
        rank: Process rank for distributed training
        world_size: Total number of processes for distributed training
    
    Returns:
        DataLoader instance
    """
    dataset = ParquetTextDataset(
        tokenizer=tokenizer,
        data_dirs=data_dirs,
        seq_len=seq_len,
        logger=logger,
        rank=rank,
        world_size=world_size
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Drop incomplete batches for stable training
    )
