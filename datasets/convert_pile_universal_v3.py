#!/usr/bin/env python3
"""
Universal Pile dataset converter v3 with improved quality filters and statistics.
"""

import os
import re
import time
import json
import tarfile
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, NamedTuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
import argparse
from tqdm import tqdm
import zstandard as zstd
import logging
import unicodedata
import ftfy
from multiprocessing import cpu_count

logging.basicConfig(level=logging.INFO)

@dataclass
class QualityStats:
    """Track quality filtering statistics."""
    total_processed: int = 0
    saved: int = 0
    filtered: int = 0
    too_short: int = 0
    too_few_unique_words: int = 0
    low_alphanumeric: int = 0
    empty_after_cleaning: int = 0
    error: int = 0
    
    def add_filtered(self, reason: str):
        self.filtered += 1
        if reason == 'too_short':
            self.too_short += 1
        elif reason == 'too_few_unique_words':
            self.too_few_unique_words += 1
        elif reason == 'low_alphanumeric':
            self.low_alphanumeric += 1
        elif reason == 'empty':
            self.empty_after_cleaning += 1
        elif reason == 'error':
            self.error += 1
    
    def add_saved(self):
        self.saved += 1
    
    def print_summary(self):
        if self.total_processed == 0:
            return
        
        save_rate = (self.saved / self.total_processed) * 100
        print(f"\n  Quality Filter Summary:")
        print(f"    Processed: {self.total_processed:,}")
        print(f"    Saved: {self.saved:,} ({save_rate:.1f}%)")
        print(f"    Filtered: {self.filtered:,}")
        
        if self.filtered > 0:
            print(f"    Filter reasons:")
            if self.too_short > 0:
                print(f"      - Too short: {self.too_short:,} ({self.too_short/self.filtered*100:.1f}%)")
            if self.too_few_unique_words > 0:
                print(f"      - Too few unique words: {self.too_few_unique_words:,} ({self.too_few_unique_words/self.filtered*100:.1f}%)")
            if self.low_alphanumeric > 0:
                print(f"      - Low alphanumeric ratio: {self.low_alphanumeric:,} ({self.low_alphanumeric/self.filtered*100:.1f}%)")
            if self.empty_after_cleaning > 0:
                print(f"      - Empty after cleaning: {self.empty_after_cleaning:,} ({self.empty_after_cleaning/self.filtered*100:.1f}%)")
            if self.error > 0:
                print(f"      - Processing errors: {self.error:,} ({self.error/self.filtered*100:.1f}%)")

class TextProcessor:
    """Text processing with quality filtering."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text while preserving meaningful content."""
        # Remove ANSI escape sequences (comprehensive)
        # This pattern matches all ANSI control sequences
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
        text = re.sub(r'\x1b\].*?\x07', '', text)  # OSC sequences
        text = re.sub(r'\x1b[PX^_].*?\x1b\\', '', text)  # DCS/PM/APC sequences
        text = re.sub(r'\x1b[NO]', '', text)  # SS2/SS3 sequences
        text = re.sub(r'\x1b[()][A-Z0-9]', '', text)  # Character set sequences
        text = re.sub(r'\x1b[<=>]', '', text)  # Other single-character sequences
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)  # Control characters (except \t, \n, \r)
        
        # Fix unicode issues
        text = ftfy.fix_text(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        # Replace multiple newlines with double newline
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        # Replace multiple spaces with single space
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        # Strip each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def check_quality(text: str, min_length: int = 100, min_unique_words: int = 20, 
                     min_alnum_ratio: float = 0.25) -> Tuple[bool, Optional[str]]:
        """Check if text meets quality standards and return reason if not."""
        # Check minimum length
        if len(text) < min_length:
            return False, 'too_short'
        
        # Check for minimum unique words
        words = text.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            if unique_words < min_unique_words:
                return False, 'too_few_unique_words'
        else:
            return False, 'empty'
        
        # Check for minimum alphanumeric content
        alnum_chars = sum(1 for c in text if c.isalnum())
        if len(text) > 0 and alnum_chars / len(text) < min_alnum_ratio:
            return False, 'low_alphanumeric'
        
        return True, None

# Process function parameters
ProcessParams = NamedTuple('ProcessParams', [
    ('file_path', Path),
    ('output_file', Path),
    ('quality_params', dict)
])

def process_text_file(params: ProcessParams) -> Tuple[int, QualityStats]:
    """Process a single text file with quality filtering."""
    stats = QualityStats()
    stats.total_processed = 1
    
    try:
        with open(params.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        cleaned_text = TextProcessor.clean_text(text)
        
        if not cleaned_text:
            stats.add_filtered('empty')
            return 0, stats
        
        # Check quality with dataset-specific parameters
        passed, reason = TextProcessor.check_quality(
            cleaned_text,
            **params.quality_params
        )
        
        if passed:
            df = pd.DataFrame({'text': [cleaned_text]})
            df.to_parquet(params.output_file, compression='snappy', index=False)
            stats.add_saved()
            return 1, stats
        else:
            stats.add_filtered(reason)
            return 0, stats
            
    except Exception as e:
        stats.add_filtered('error')
        return 0, stats

def process_jsonl_chunk(args) -> Tuple[int, QualityStats]:
    """Process a chunk of JSONL data."""
    chunk_data, output_file, dataset_name, quality_params = args
    stats = QualityStats()
    
    texts = []
    for line in chunk_data:
        if line.strip():
            stats.total_processed += 1
            try:
                data = json.loads(line)
                text = data.get('text', '')
                if text:
                    cleaned_text = TextProcessor.clean_text(text)
                    
                    if not cleaned_text:
                        stats.add_filtered('empty')
                        continue
                    
                    passed, reason = TextProcessor.check_quality(
                        cleaned_text,
                        **quality_params
                    )
                    
                    if passed:
                        texts.append(cleaned_text)
                        stats.add_saved()
                    else:
                        stats.add_filtered(reason)
            except:
                stats.add_filtered('error')
    
    if texts:
        df = pd.DataFrame({'text': texts})
        df.to_parquet(output_file, compression='snappy', index=False)
        return len(texts), stats
    return 0, stats

class UniversalPileConverterV3:
    """Universal converter with improved quality filtering and statistics."""
    
    def __init__(self, input_dir: Path, output_dir: Path, num_workers: Optional[int] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers or cpu_count()
        self.temp_base = None
        
        # Dataset patterns
        self.dataset_patterns = {
            'europarl': 'EuroParliamentProceedings*.jsonl.zst',
            'freelaw': 'FreeLaw*.jsonl.zst',
            'hackernews': 'HackerNews*.jsonl.zst',
            'nih_exporter': 'NIH_ExPORTER*.jsonl.zst',
            'pubmed': 'PubMed*.jsonl.zst',
            'wikipedia': 'Wikipedia*.jsonl.zst',
            'youtubesubtitles': 'YoutubeSubtitles*.jsonl.zst',
            'literotica': 'Literotica*.jsonl.zst',
            'gutenberg': 'Gutenberg*.jsonl.zst',
            'bookscorpus': 'BookCorpus*.jsonl.zst',
            'opensubtitles': 'OpenSubtitles*.jsonl.zst',
            'openwebtext': 'OpenWebText*.jsonl.zst',
            'pmc_extracts': 'PMC_extracts*.tar.gz',
            'books1': 'books1*.tar.gz',
            'books3': 'books3*.tar.gz',
            'github': 'github.tar',
            'stackexchange': 'stackexchange*.tar',
        }
        
        # Dataset-specific quality parameters
        self.quality_params = {
            # Books: Allow more repetition, require longer texts
            'books1': {'min_length': 1000, 'min_unique_words': 50, 'min_alnum_ratio': 0.4},
            'books3': {'min_length': 1000, 'min_unique_words': 50, 'min_alnum_ratio': 0.4},
            'literotica': {'min_length': 50, 'min_unique_words': 10, 'min_alnum_ratio': 0.3},
            'gutenberg': {'min_length': 1000, 'min_unique_words': 50, 'min_alnum_ratio': 0.4},
            'bookscorpus': {'min_length': 1000, 'min_unique_words': 50, 'min_alnum_ratio': 0.4},
            
            # Academic/technical: Medium requirements
            'pmc_extracts': {'min_length': 200, 'min_unique_words': 30, 'min_alnum_ratio': 0.3},
            'pubmed': {'min_length': 100, 'min_unique_words': 20, 'min_alnum_ratio': 0.3},
            'nih_exporter': {'min_length': 50, 'min_unique_words': 10, 'min_alnum_ratio': 0.3},
            'wikipedia': {'min_length': 200, 'min_unique_words': 30, 'min_alnum_ratio': 0.4},
            
            # Code/technical: Lower alphanumeric requirements
            'github': {'min_length': 50, 'min_unique_words': 10, 'min_alnum_ratio': 0.2},
            'stackexchange': {'min_length': 50, 'min_unique_words': 15, 'min_alnum_ratio': 0.3},
            
            # Conversational/subtitles: Very minimal requirements
            'opensubtitles': {'min_length': 20, 'min_unique_words': 5, 'min_alnum_ratio': 0.2},
            'youtubesubtitles': {'min_length': 20, 'min_unique_words': 5, 'min_alnum_ratio': 0.2},
            
            # General web text
            'openwebtext': {'min_length': 100, 'min_unique_words': 20, 'min_alnum_ratio': 0.3},
            'hackernews': {'min_length': 50, 'min_unique_words': 15, 'min_alnum_ratio': 0.3},
            
            # Legal text: Can be repetitive
            'freelaw': {'min_length': 200, 'min_unique_words': 30, 'min_alnum_ratio': 0.4},
            
            # Parliamentary proceedings: Can be repetitive
            'europarl': {'min_length': 100, 'min_unique_words': 20, 'min_alnum_ratio': 0.4},
            
            # Default for unknown datasets
            'default': {'min_length': 100, 'min_unique_words': 20, 'min_alnum_ratio': 0.25}
        }
    
    def get_quality_params(self, dataset_name: str) -> dict:
        """Get quality parameters for a dataset."""
        return self.quality_params.get(dataset_name, self.quality_params['default'])
    
    def setup_temp_dir(self):
        """Create temporary directory for decompression."""
        # Create temp directory under /mnt/training/ instead of /tmp
        training_temp_dir = Path('/mnt/training/temp')
        training_temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_base = tempfile.mkdtemp(prefix='pile_convert_', dir=str(training_temp_dir))
        print(f"📁 Temporary directory: {self.temp_base}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_base and os.path.exists(self.temp_base):
            print(f"\n🧹 Cleaning up {self.temp_base}...")
            shutil.rmtree(self.temp_base, ignore_errors=True)
    
    def get_dataset_name(self, file_path: Path) -> Optional[str]:
        """Identify dataset name from file path."""
        for dataset, pattern in self.dataset_patterns.items():
            if file_path.match(pattern):
                return dataset
        return None
    
    def process_jsonl_file(self, jsonl_path: Path, dataset_name: str) -> Tuple[int, QualityStats]:
        """Process a JSONL file with quality filtering."""
        output_dir = self.output_dir / dataset_name / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        quality_params = self.get_quality_params(dataset_name)
        
        # Count lines for progress
        print(f"📊 Counting lines in {jsonl_path.name}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"📄 Processing {total_lines:,} lines...")
        
        # Process in chunks
        chunk_size = 10000
        chunk_num = 0
        total_texts = 0
        aggregate_stats = QualityStats()
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            chunk = []
            work_items = []
            
            for line in tqdm(f, total=total_lines, desc="Reading"):
                chunk.append(line)
                
                if len(chunk) >= chunk_size:
                    output_file = output_dir / f"chunk_{chunk_num:04d}.parquet"
                    work_items.append((chunk, output_file, dataset_name, quality_params))
                    chunk = []
                    chunk_num += 1
            
            # Don't forget the last chunk
            if chunk:
                output_file = output_dir / f"chunk_{chunk_num:04d}.parquet"
                work_items.append((chunk, output_file, dataset_name, quality_params))
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_jsonl_chunk, item): i 
                      for i, item in enumerate(work_items)}
            
            with tqdm(total=len(work_items), desc=f"Processing {dataset_name} chunks") as pbar:
                for future in as_completed(futures):
                    try:
                        count, stats = future.result()
                        total_texts += count
                        
                        # Aggregate statistics
                        aggregate_stats.total_processed += stats.total_processed
                        aggregate_stats.saved += stats.saved
                        aggregate_stats.filtered += stats.filtered
                        aggregate_stats.too_short += stats.too_short
                        aggregate_stats.too_few_unique_words += stats.too_few_unique_words
                        aggregate_stats.low_alphanumeric += stats.low_alphanumeric
                        aggregate_stats.empty_after_cleaning += stats.empty_after_cleaning
                        aggregate_stats.error += stats.error
                        
                        pbar.set_postfix({'texts': f"{total_texts:,}"})
                    except Exception as e:
                        print(f"\n❌ Error processing chunk: {e}")
                    pbar.update(1)
        
        # Merge small files
        self.merge_chunks(output_dir)
        
        return total_texts, aggregate_stats
    
    def process_tar_with_text_files(self, tar_path: Path, dataset_name: str) -> Tuple[int, QualityStats]:
        """Process TAR archive containing text files."""
        output_dir = self.output_dir / dataset_name / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        quality_params = self.get_quality_params(dataset_name)
        
        temp_extract = Path(self.temp_base) / f"{dataset_name}_extract"
        temp_extract.mkdir(exist_ok=True)
        
        print(f"📂 Extracting text files from {tar_path.name}...")
        text_files = []
        
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting"):
                if member.isfile() and (member.name.endswith('.txt') or member.name.endswith('.md')):
                    tar.extract(member, temp_extract)
                    text_files.append(temp_extract / member.name)
        
        print(f"📄 Found {len(text_files)} text files")
        
        # Process text files in parallel
        work_items = []
        for f in text_files:
            # Create unique output name
            output_name = f.name
            if output_name.endswith('.epub.txt'):
                output_name = output_name[:-9]
            elif output_name.endswith('.txt'):
                output_name = output_name[:-4]
            elif output_name.endswith('.md'):
                output_name = output_name[:-3]
            
            output_file = output_dir / f"{output_name}.parquet"
            params = ProcessParams(f, output_file, quality_params)
            work_items.append(params)
        
        total_texts = 0
        aggregate_stats = QualityStats()
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_text_file, item): i 
                      for i, item in enumerate(work_items)}
            
            with tqdm(total=len(work_items), desc=f"Processing {dataset_name} texts") as pbar:
                for future in as_completed(futures):
                    try:
                        count, stats = future.result()
                        total_texts += count
                        
                        # Aggregate statistics
                        aggregate_stats.total_processed += stats.total_processed
                        aggregate_stats.saved += stats.saved
                        aggregate_stats.filtered += stats.filtered
                        aggregate_stats.too_short += stats.too_short
                        aggregate_stats.too_few_unique_words += stats.too_few_unique_words
                        aggregate_stats.low_alphanumeric += stats.low_alphanumeric
                        aggregate_stats.empty_after_cleaning += stats.empty_after_cleaning
                        aggregate_stats.error += stats.error
                        
                        pbar.set_postfix({'saved': f"{aggregate_stats.saved:,}"})
                    except Exception as e:
                        print(f"\n❌ Error processing text: {e}")
                        aggregate_stats.add_filtered('error')
                    pbar.update(1)
        
        # Clean up extracted files
        shutil.rmtree(temp_extract, ignore_errors=True)
        
        # Merge small files
        self.merge_chunks(output_dir)
        
        return total_texts, aggregate_stats
    
    def process_tar_with_jsonl_files(self, tar_path: Path, dataset_name: str) -> Tuple[int, QualityStats]:
        """Process TAR archive containing JSONL.zst files (like GitHub dataset)."""
        output_dir = self.output_dir / dataset_name / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        quality_params = self.get_quality_params(dataset_name)
        
        temp_extract = Path(self.temp_base) / f"{dataset_name}_extract"
        temp_extract.mkdir(exist_ok=True)
        
        print(f"📂 Extracting JSONL files from {tar_path.name}...")
        jsonl_files = []
        
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting"):
                if member.isfile() and member.name.endswith('.jsonl.zst'):
                    tar.extract(member, temp_extract)
                    jsonl_files.append(temp_extract / member.name)
        
        print(f"📄 Found {len(jsonl_files)} JSONL.zst files")
        
        total_texts = 0
        aggregate_stats = QualityStats()
        
        # Process each JSONL file
        for jsonl_file in tqdm(jsonl_files, desc=f"Processing {dataset_name} JSONL files"):
            try:
                # Decompress the .zst file
                decompressed_path = self.decompress_file(jsonl_file, temp_extract)
                
                # Process the JSONL file
                texts, stats = self.process_jsonl_file(decompressed_path, dataset_name)
                total_texts += texts
                
                # Aggregate statistics
                aggregate_stats.total_processed += stats.total_processed
                aggregate_stats.saved += stats.saved
                aggregate_stats.filtered += stats.filtered
                aggregate_stats.too_short += stats.too_short
                aggregate_stats.too_few_unique_words += stats.too_few_unique_words
                aggregate_stats.low_alphanumeric += stats.low_alphanumeric
                aggregate_stats.empty_after_cleaning += stats.empty_after_cleaning
                aggregate_stats.error += stats.error
                
                # Clean up decompressed file
                decompressed_path.unlink(missing_ok=True)
                
            except Exception as e:
                print(f"\n❌ Error processing {jsonl_file.name}: {e}")
                aggregate_stats.add_filtered('error')
        
        # Clean up extracted files
        shutil.rmtree(temp_extract, ignore_errors=True)
        
        # Merge small files
        self.merge_chunks(output_dir)
        
        return total_texts, aggregate_stats
    
    def merge_chunks(self, output_dir: Path, target_size_mb: int = 256):
        """Merge small parquet files into larger ones."""
        chunk_files = sorted(output_dir.glob("chunk_*.parquet"))
        if len(chunk_files) <= 1:
            return
        
        print(f"🔄 Merging {len(chunk_files)} chunks...")
        merged_count = 0
        current_batch = []
        current_size = 0
        
        for chunk_file in chunk_files:
            file_size = chunk_file.stat().st_size
            
            if current_size + file_size > target_size_mb * 1024 * 1024 and current_batch:
                # Write merged file
                merged_df = pd.concat([pd.read_parquet(f) for f in current_batch])
                output_file = output_dir / f"part_{merged_count:04d}.parquet"
                merged_df.to_parquet(output_file, compression='snappy', index=False)
                
                # Clean up chunk files
                for f in current_batch:
                    f.unlink()
                
                merged_count += 1
                current_batch = [chunk_file]
                current_size = file_size
            else:
                current_batch.append(chunk_file)
                current_size += file_size
        
        # Handle remaining files
        if current_batch:
            merged_df = pd.concat([pd.read_parquet(f) for f in current_batch])
            output_file = output_dir / f"part_{merged_count:04d}.parquet"
            merged_df.to_parquet(output_file, compression='snappy', index=False)
            
            for f in current_batch:
                f.unlink()
    
    def process_dataset(self, file_path: Path, dataset_name: str) -> Tuple[str, float, int, QualityStats]:
        """Process a single dataset file."""
        print(f"\n{'='*60}")
        print(f"📥 Processing {dataset_name}: {file_path.name}")
        print(f"   Size: {file_path.stat().st_size / 1e9:.1f} GB")
        
        quality_params = self.get_quality_params(dataset_name)
        print(f"   Quality params: min_length={quality_params['min_length']}, "
              f"min_unique_words={quality_params['min_unique_words']}, "
              f"min_alnum_ratio={quality_params['min_alnum_ratio']}")
        
        start_time = time.time()
        temp_dir = Path(self.temp_base) / dataset_name
        temp_dir.mkdir(exist_ok=True)
        
        total_texts = 0
        stats = QualityStats()
        
        try:
            # Decompress if needed
            decompressed_path = file_path
            if file_path.suffix == '.zst':
                decompressed_path = self.decompress_file(file_path, temp_dir)
            
            # Process based on file type
            if file_path.name.endswith('.jsonl.zst') or decompressed_path.suffix == '.jsonl':
                total_texts, stats = self.process_jsonl_file(decompressed_path, dataset_name)
            elif file_path.suffix in ['.tar', '.gz']:
                if file_path.suffix == '.gz':
                    # Decompress .tar.gz
                    import gzip
                    tar_path = temp_dir / file_path.stem
                    with gzip.open(file_path, 'rb') as gz_in:
                        with open(tar_path, 'wb') as tar_out:
                            shutil.copyfileobj(gz_in, tar_out)
                    decompressed_path = tar_path
                
                # Check if it's a text archive or nested archive with JSONL files
                with tarfile.open(decompressed_path, 'r') as tar:
                    members = tar.getmembers()[:10]  # Check first 10 members
                    has_text_files = any(m.name.endswith(('.txt', '.md')) for m in members)
                    has_jsonl_files = any(m.name.endswith('.jsonl.zst') for m in members)
                    
                    if has_jsonl_files:
                        total_texts, stats = self.process_tar_with_jsonl_files(decompressed_path, dataset_name)
                    elif has_text_files:
                        total_texts, stats = self.process_tar_with_text_files(decompressed_path, dataset_name)
            
            # Clean up temp files
            if decompressed_path != file_path:
                decompressed_path.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
        
        elapsed = time.time() - start_time
        
        # Print summary
        print(f"\n✅ Completed {dataset_name} in {elapsed/60:.1f} minutes")
        print(f"   Total texts saved: {total_texts:,}")
        
        # Print quality statistics
        stats.print_summary()
        
        # Calculate and show final statistics
        output_dir = self.output_dir / dataset_name / 'data'
        if output_dir.exists():
            parquet_files = list(output_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in parquet_files)
            print(f"\n   Output statistics:")
            print(f"     Parquet files: {len(parquet_files):,}")
            print(f"     Total size: {total_size / 1e9:.2f} GB")
            if total_texts > 0:
                print(f"     Avg text size: {total_size / total_texts / 1024:.1f} KB")
        
        return dataset_name, elapsed, total_texts, stats
    
    def decompress_file(self, file_path: Path, temp_dir: Path) -> Path:
        """Decompress a file and return path to decompressed content."""
        output_path = temp_dir / file_path.stem
        
        print(f"📦 Decompressing {file_path.name}...")
        start_time = time.time()
        
        if file_path.suffix == '.zst':
            dctx = zstd.ZstdDecompressor()
            with open(file_path, 'rb') as ifh, open(output_path, 'wb') as ofh:
                dctx.copy_stream(ifh, ofh)
        
        elapsed = time.time() - start_time
        print(f"✅ Decompressed in {elapsed:.1f}s")
        return output_path
    
    def process_custom_directory(self, custom_dir: Path) -> Tuple[int, QualityStats]:
        """Process all text files from a custom directory into a 'self' dataset."""
        dataset_name = 'self'
        output_dir = self.output_dir / dataset_name / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use relaxed quality parameters for code/documentation
        quality_params = {
            'min_length': 50,
            'min_unique_words': 10,
            'min_alnum_ratio': 0.2
        }
        
        # Common text file extensions
        text_extensions = {
            '.py', '.md', '.txt', '.sh', '.log', '.json', '.yaml', '.yml',
            '.toml', '.ini', '.cfg', '.conf', '.js', '.ts', '.jsx', '.tsx',
            '.java', '.c', '.cpp', '.h', '.hpp', '.rs', '.go', '.rb',
            '.php', '.html', '.css', '.xml', '.sql', '.r', '.m', '.swift',
            '.kt', '.scala', '.jl', '.lua', '.pl', '.ps1', '.bat', '.cmd',
            '.dockerfile', '.makefile', '.cmake', '.gradle', '.maven'
        }
        
        # Find all text files recursively
        print(f"🔍 Scanning {custom_dir} for text files...")
        text_files = []
        
        for ext in text_extensions:
            text_files.extend(custom_dir.rglob(f'*{ext}'))
        
        # Also include files with no extension that might be text
        for file in custom_dir.rglob('*'):
            if file.is_file() and not file.suffix and file.name not in ['.git', '.svn']:
                # Check if it's a text file by trying to read first few bytes
                try:
                    with open(file, 'rb') as f:
                        chunk = f.read(512)
                        if all(b < 128 or b in [9, 10, 13] for b in chunk):  # ASCII + tab/newline/return
                            text_files.append(file)
                except:
                    pass
        
        # Remove duplicates and filter out binary files and hidden directories
        text_files = list(set(text_files))
        text_files = [f for f in text_files if not any(part.startswith('.') for part in f.parts[len(custom_dir.parts):])]
        
        print(f"📄 Found {len(text_files)} text files")
        
        # Process files in parallel
        work_items = []
        for i, file_path in enumerate(text_files):
            # Create unique output name based on relative path
            rel_path = file_path.relative_to(custom_dir)
            safe_name = str(rel_path).replace('/', '_').replace('\\', '_')
            output_file = output_dir / f"{safe_name}_{i:06d}.parquet"
            params = ProcessParams(file_path, output_file, quality_params)
            work_items.append(params)
        
        total_texts = 0
        aggregate_stats = QualityStats()
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_text_file, item): i 
                      for i, item in enumerate(work_items)}
            
            with tqdm(total=len(work_items), desc=f"Processing {dataset_name} files") as pbar:
                for future in as_completed(futures):
                    try:
                        count, stats = future.result()
                        total_texts += count
                        
                        # Aggregate statistics
                        aggregate_stats.total_processed += stats.total_processed
                        aggregate_stats.saved += stats.saved
                        aggregate_stats.filtered += stats.filtered
                        aggregate_stats.too_short += stats.too_short
                        aggregate_stats.too_few_unique_words += stats.too_few_unique_words
                        aggregate_stats.low_alphanumeric += stats.low_alphanumeric
                        aggregate_stats.empty_after_cleaning += stats.empty_after_cleaning
                        aggregate_stats.error += stats.error
                        
                        pbar.set_postfix({'saved': f"{aggregate_stats.saved:,}"})
                    except Exception as e:
                        print(f"\n❌ Error processing file: {e}")
                        aggregate_stats.add_filtered('error')
                    pbar.update(1)
        
        # Don't merge files for custom directories - preserve 1:1 mapping
        # self.merge_chunks(output_dir)
        
        return total_texts, aggregate_stats
    
    def convert(self, datasets: Optional[List[str]] = None, custom_dir: Optional[Path] = None):
        """Convert specified datasets or all available."""
        self.setup_temp_dir()
        
        try:
            # Handle custom directory if specified
            if custom_dir:
                print(f"\n📂 Processing custom directory: {custom_dir}")
                print(f"   Creating dataset: 'self'")
                
                start_time = time.time()
                total_texts, stats = self.process_custom_directory(custom_dir)
                elapsed = time.time() - start_time
                
                print(f"\n✅ Completed 'self' dataset in {elapsed/60:.1f} minutes")
                print(f"   Total texts saved: {total_texts:,}")
                stats.print_summary()
                
                # Show output statistics
                output_dir = self.output_dir / 'self' / 'data'
                if output_dir.exists():
                    parquet_files = list(output_dir.glob("*.parquet"))
                    total_size = sum(f.stat().st_size for f in parquet_files)
                    print(f"\n   Output statistics:")
                    print(f"     Parquet files: {len(parquet_files):,}")
                    print(f"     Total size: {total_size / 1e6:.2f} MB")
                    if total_texts > 0:
                        print(f"     Avg text size: {total_size / total_texts / 1024:.1f} KB")
                
                return
            
            # Find available dataset files
            available_files = []
            for dataset, pattern in self.dataset_patterns.items():
                if datasets and dataset not in datasets:
                    continue
                
                files = list(self.input_dir.glob(pattern))
                for f in files:
                    available_files.append((f, dataset))
            
            if not available_files:
                print("❌ No matching dataset files found!")
                return
            
            print(f"\n📊 Found {len(available_files)} datasets to process")
            
            # Process each dataset
            results = []
            total_stats = QualityStats()
            
            for file_path, dataset_name in available_files:
                dataset, elapsed, texts, stats = self.process_dataset(file_path, dataset_name)
                results.append((dataset, elapsed, texts))
                
                # Aggregate overall statistics
                total_stats.total_processed += stats.total_processed
                total_stats.saved += stats.saved
                total_stats.filtered += stats.filtered
                total_stats.too_short += stats.too_short
                total_stats.too_few_unique_words += stats.too_few_unique_words
                total_stats.low_alphanumeric += stats.low_alphanumeric
                total_stats.empty_after_cleaning += stats.empty_after_cleaning
                total_stats.error += stats.error
            
            # Print final summary
            print(f"\n{'='*60}")
            print("🎉 Conversion Complete!")
            print(f"\nDataset Summary:")
            total_time = 0
            total_texts = 0
            for dataset, elapsed, texts in results:
                print(f"  {dataset}: {texts:,} texts in {elapsed/60:.1f} minutes")
                total_time += elapsed
                total_texts += texts
            
            print(f"\nTotal: {total_texts:,} texts in {total_time/60:.1f} minutes")
            
            # Print overall quality statistics
            print(f"\n📊 Overall Quality Statistics:")
            total_stats.print_summary()
            
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Convert Pile datasets to Parquet format')
    parser.add_argument('--input-dir', type=str, required=False,
                        help='Input directory containing Pile datasets')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for Parquet files')
    parser.add_argument('--datasets', type=str, nargs='*',
                        help='Specific datasets to convert (default: all)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('--custom-dir', type=str,
                        help='Process all text files from a custom directory into "self" dataset')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.custom_dir and not args.input_dir:
        parser.error("Either --input-dir or --custom-dir must be specified")
    
    if args.custom_dir and args.input_dir:
        parser.error("Cannot specify both --input-dir and --custom-dir")
    
    if args.custom_dir and args.datasets:
        parser.error("--datasets cannot be used with --custom-dir")
    
    # Create converter with input_dir even if using custom_dir (required by constructor)
    converter = UniversalPileConverterV3(
        Path(args.input_dir) if args.input_dir else Path('.'),
        Path(args.output_dir),
        args.num_workers
    )
    
    # Convert either custom directory or standard datasets
    if args.custom_dir:
        converter.convert(custom_dir=Path(args.custom_dir))
    else:
        converter.convert(args.datasets)

if __name__ == '__main__':
    main()
