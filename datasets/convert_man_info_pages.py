#!/usr/bin/env python3
"""
Convert local man and info pages to a Parquet dataset.
Extracts and processes all manual pages and GNU info documentation.
"""

import os
import re
import gzip
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import argparse
from tqdm import tqdm
import logging
import unicodedata
import ftfy

logging.basicConfig(level=logging.INFO)

class ManPageProcessor:
    """Process man pages into clean text."""
    
    @staticmethod
    def find_man_pages() -> List[Path]:
        """Find all man pages on the system."""
        man_paths = [
            "/usr/share/man",
            "/usr/local/share/man",
            "/usr/local/man",
            "/opt/man",
            "/usr/X11R6/man"
        ]
        
        man_files = []
        for base_path in man_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith(('.gz', '.bz2')) or re.match(r'.*\.\d+$', file):
                            man_files.append(Path(root) / file)
        
        return man_files
    
    @staticmethod
    def extract_man_page(file_path: Path) -> Optional[str]:
        """Extract and format a man page to plain text."""
        try:
            # Use man command to properly format the page
            if file_path.suffix == '.gz':
                # For compressed files, we need to extract first
                cmd = ['zcat', str(file_path), '|', 'groff', '-mandoc', '-Tascii', '|', 'col', '-b']
                result = subprocess.run(' '.join(cmd), shell=True, capture_output=True, text=True)
            else:
                # For uncompressed files
                cmd = ['groff', '-mandoc', '-Tascii', str(file_path), '|', 'col', '-b']
                result = subprocess.run(' '.join(cmd), shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                return result.stdout
            
            # Fallback: try using man directly
            # Extract the page name from filename
            page_name = file_path.stem
            if page_name.endswith('.gz'):
                page_name = Path(page_name).stem
            
            # Try to determine section from path
            section_match = re.search(r'/man(\d+)/', str(file_path))
            if section_match:
                section = section_match.group(1)
                cmd = ['man', section, page_name]
            else:
                cmd = ['man', page_name]
            
            result = subprocess.run(cmd, capture_output=True, text=True, env={'MANPATH': str(file_path.parent.parent)})
            if result.returncode == 0:
                return result.stdout
                
        except Exception as e:
            logging.debug(f"Error processing {file_path}: {e}")
        
        return None
    
    @staticmethod
    def clean_man_text(text: str) -> str:
        """Clean man page text."""
        # Remove backspace-based formatting
        text = re.sub(r'.\x08', '', text)
        
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
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove form feed characters
        text = text.replace('\f', '\n')
        
        return text.strip()


class InfoPageProcessor:
    """Process GNU info pages into clean text."""
    
    @staticmethod
    def find_info_pages() -> List[Path]:
        """Find all info pages on the system."""
        info_paths = [
            "/usr/share/info",
            "/usr/local/share/info",
            "/opt/info"
        ]
        
        info_files = []
        for base_path in info_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith(('.info', '.info.gz', '.info.bz2')):
                            info_files.append(Path(root) / file)
        
        return info_files
    
    @staticmethod
    def extract_info_page(file_path: Path) -> Optional[str]:
        """Extract info page to plain text."""
        try:
            # Use info command to extract text
            cmd = ['info', '--subnodes', '-f', str(file_path), '-o', '-']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                return result.stdout
                
            # Fallback: try to read directly
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
        except Exception as e:
            logging.debug(f"Error processing {file_path}: {e}")
        
        return None
    
    @staticmethod
    def clean_info_text(text: str) -> str:
        """Clean info page text."""
        # Remove info navigation markers
        text = re.sub(r'\x1f\n(File:|Tag:|Node:|Ref:)[^\n]*\n', '', text)
        
        # Remove byte offsets
        text = re.sub(r'\x7f[^\n]*\n', '', text)
        
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
        
        # Clean up formatting
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()


def process_man_file(file_path: Path) -> Optional[Dict[str, str]]:
    """Process a single man page file."""
    text = ManPageProcessor.extract_man_page(file_path)
    if text:
        cleaned_text = ManPageProcessor.clean_man_text(text)
        if len(cleaned_text) > 100:  # Minimum length filter
            # Extract metadata
            page_name = file_path.stem
            if page_name.endswith('.gz'):
                page_name = Path(page_name).stem
            
            section_match = re.search(r'/man(\d+)/', str(file_path))
            section = section_match.group(1) if section_match else 'unknown'
            
            return {
                'text': cleaned_text,
                'source': 'man_page',
                'page_name': page_name,
                'section': section,
                'file_path': str(file_path)
            }
    return None


def process_info_file(file_path: Path) -> Optional[Dict[str, str]]:
    """Process a single info file."""
    text = InfoPageProcessor.extract_info_page(file_path)
    if text:
        cleaned_text = InfoPageProcessor.clean_info_text(text)
        if len(cleaned_text) > 100:  # Minimum length filter
            page_name = file_path.stem
            if page_name.endswith('.info'):
                page_name = Path(page_name).stem
            
            return {
                'text': cleaned_text,
                'source': 'info_page',
                'page_name': page_name,
                'section': 'info',
                'file_path': str(file_path)
            }
    return None


class ManInfoConverter:
    """Main converter for man and info pages."""
    
    def __init__(self, output_dir: Path, num_workers: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers or cpu_count()
    
    def convert(self):
        """Convert all man and info pages to Parquet format."""
        print("\n📚 Converting Man and Info Pages to Dataset")
        print(f"   Output: {self.output_dir}")
        print(f"   Workers: {self.num_workers}")
        
        # Create output directory
        output_data_dir = self.output_dir / 'man_info_pages' / 'data'
        output_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all man pages
        print("\n🔍 Searching for man pages...")
        man_files = ManPageProcessor.find_man_pages()
        print(f"   Found {len(man_files):,} man page files")
        
        # Find all info pages
        print("\n🔍 Searching for info pages...")
        info_files = InfoPageProcessor.find_info_pages()
        print(f"   Found {len(info_files):,} info page files")
        
        all_documents = []
        
        # Process man pages
        if man_files:
            print("\n📖 Processing man pages...")
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(process_man_file, f): f for f in man_files}
                
                with tqdm(total=len(man_files), desc="Processing man pages") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                all_documents.append(result)
                        except Exception as e:
                            logging.debug(f"Error: {e}")
                        pbar.update(1)
        
        # Process info pages
        if info_files:
            print("\n📖 Processing info pages...")
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(process_info_file, f): f for f in info_files}
                
                with tqdm(total=len(info_files), desc="Processing info pages") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                all_documents.append(result)
                        except Exception as e:
                            logging.debug(f"Error: {e}")
                        pbar.update(1)
        
        # Save to Parquet
        if all_documents:
            print(f"\n💾 Saving {len(all_documents):,} documents to Parquet...")
            
            # Split into multiple files if too large
            docs_per_file = 5000
            file_count = 0
            
            for i in range(0, len(all_documents), docs_per_file):
                batch = all_documents[i:i + docs_per_file]
                df = pd.DataFrame(batch)
                
                output_file = output_data_dir / f"part_{file_count:04d}.parquet"
                df.to_parquet(output_file, compression='snappy', index=False)
                
                file_count += 1
            
            print(f"✅ Saved {file_count} Parquet files")
            
            # Print summary statistics
            print(f"\n📊 Summary:")
            print(f"   Total documents: {len(all_documents):,}")
            
            # Count by source
            df_all = pd.DataFrame(all_documents)
            source_counts = df_all['source'].value_counts()
            for source, count in source_counts.items():
                print(f"   {source}: {count:,}")
            
            # Count man pages by section
            man_pages = df_all[df_all['source'] == 'man_page']
            if not man_pages.empty:
                section_counts = man_pages['section'].value_counts()
                print(f"\n   Man pages by section:")
                for section, count in section_counts.items():
                    print(f"     Section {section}: {count:,}")
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in output_data_dir.glob("*.parquet"))
            print(f"\n   Total size: {total_size / 1e6:.1f} MB")
            
        else:
            print("\n❌ No documents were successfully processed")


def main():
    parser = argparse.ArgumentParser(description='Convert man and info pages to Parquet dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for Parquet files')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    
    args = parser.parse_args()
    
    converter = ManInfoConverter(
        Path(args.output_dir),
        args.num_workers
    )
    
    converter.convert()


if __name__ == '__main__':
    main()
