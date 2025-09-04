"""
AMR Processor Module
===================

This module handles AMR data cleaning, preprocessing, and conversion to training format.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AMRSample:
    """Data class representing a single AMR sample."""
    sentence: str
    amr: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "input": self.sentence,
            "output": self.amr
        }


class AMRProcessor:
    """
    Processes AMR files and converts them to training format.
    
    This class handles:
    - Cleaning and normalizing AMR files
    - Extracting sentence-AMR pairs
    - Converting to JSONL format for efficient processing
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize AMR processor.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def clean_amr_file(self, input_file_path: str, output_file_path: Optional[str] = None) -> str:
        """
        Clean and normalize AMR file by grouping sentences with their AMR representations.
        
        Args:
            input_file_path: Path to input AMR file
            output_file_path: Optional output path. If None, creates 'cleaned_amr.txt' in same directory
            
        Returns:
            Path to cleaned AMR file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If file operations fail
        """
        input_path = Path(input_file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        
        if output_file_path is None:
            output_file_path = input_path.parent / "cleaned_amr.txt"
        else:
            output_file_path = Path(output_file_path)
        
        self.logger.info(f"Cleaning AMR file: {input_file_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            blocks = self._group_amr_blocks(lines)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for block in blocks:
                    for line in block:
                        f.write(line + '\n')
                    f.write('-' * 50 + '\n')
            
            self.logger.info(f"✅ Cleaned AMR file saved to: {output_file_path}")
            return str(output_file_path)
            
        except IOError as e:
            self.logger.error(f"Error processing file: {e}")
            raise
    
    def _group_amr_blocks(self, lines: List[str]) -> List[List[str]]:
        """
        Group lines into AMR blocks based on sentence markers.
        
        Args:
            lines: List of lines from AMR file
            
        Returns:
            List of blocks, where each block contains lines for one AMR sample
        """
        blocks = []
        current_block = []
        
        for line in lines:
            line = line.rstrip()
            
            if line.startswith("#::snt"):
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            
            if line.strip():  # Skip empty lines
                current_block.append(line)
        
        # Add the last block if it exists
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def extract_sentence_amr_pairs(self, amr_file_path: str) -> List[AMRSample]:
        """
        Extract sentence-AMR pairs from cleaned AMR file.
        
        Args:
            amr_file_path: Path to cleaned AMR file
            
        Returns:
            List of AMRSample objects
            
        Raises:
            FileNotFoundError: If AMR file doesn't exist
            ValueError: If AMR format is invalid
        """
        amr_path = Path(amr_file_path)
        if not amr_path.exists():
            raise FileNotFoundError(f"AMR file not found: {amr_file_path}")
        
        self.logger.info(f"Extracting sentence-AMR pairs from: {amr_file_path}")
        
        samples = []
        
        try:
            with open(amr_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            blocks = content.split('-' * 50)
            
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                
                sample = self._parse_amr_block(block)
                if sample:
                    samples.append(sample)
            
            self.logger.info(f"✅ Extracted {len(samples)} sentence-AMR pairs")
            return samples
            
        except IOError as e:
            self.logger.error(f"Error reading AMR file: {e}")
            raise
    
    def _parse_amr_block(self, block: str) -> Optional[AMRSample]:
        """
        Parse a single AMR block to extract sentence and AMR.
        
        Args:
            block: String containing one AMR block
            
        Returns:
            AMRSample object or None if parsing fails
        """
        lines = block.strip().split('\n')
        sentence = None
        amr_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("#::snt"):
                # Extract sentence after "#::snt "
                sentence = line[6:].strip()
            elif line and not line.startswith("#"):
                # AMR content (non-comment lines)
                amr_lines.append(line)
        
        if sentence and amr_lines:
            amr = '\n'.join(amr_lines)
            return AMRSample(sentence=sentence, amr=amr)
        
        return None
    
    def save_to_jsonl(self, samples: List[AMRSample], output_path: str) -> None:
        """
        Save AMR samples to JSONL format for efficient processing.
        
        Args:
            samples: List of AMRSample objects
            output_path: Path to output JSONL file
            
        Raises:
            IOError: If file writing fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving {len(samples)} samples to JSONL: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    json.dump(sample.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            
            self.logger.info(f"✅ Saved {len(samples)} samples to JSONL format")
            
        except IOError as e:
            self.logger.error(f"Error saving JSONL file: {e}")
            raise
    
    def process_amr_files(self, input_dir: str, output_dir: str) -> str:
        """
        Process all AMR files in a directory and create training data.
        
        Args:
            input_dir: Directory containing AMR files
            output_dir: Directory to save processed data
            
        Returns:
            Path to the generated JSONL file
            
        Raises:
            FileNotFoundError: If input directory doesn't exist
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing AMR files from: {input_dir}")
        
        all_samples = []
        
        # Process all .txt files in the input directory
        for amr_file in input_path.glob("*.txt"):
            self.logger.info(f"Processing file: {amr_file.name}")
            
            # Clean the AMR file
            cleaned_file = self.clean_amr_file(str(amr_file))
            
            # Extract samples
            samples = self.extract_sentence_amr_pairs(cleaned_file)
            all_samples.extend(samples)
            
            # Clean up temporary cleaned file
            Path(cleaned_file).unlink()
        
        # Save all samples to JSONL
        output_jsonl = output_path / "amr_training_data.jsonl"
        self.save_to_jsonl(all_samples, str(output_jsonl))
        
        self.logger.info(f"✅ Processing complete. Total samples: {len(all_samples)}")
        return str(output_jsonl)
