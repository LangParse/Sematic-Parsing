"""
AMR Predictor Module
===================

This module handles model inference and prediction with clean interface.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
from transformers import T5ForConditionalGeneration
import re

from ..tokenization import ViT5Tokenizer
from ..utils import setup_logger


class AMRPredictor:
    """
    Clean interface for AMR prediction and inference.
    
    This class provides:
    - Simple prediction interface
    - Batch processing capabilities
    - AMR formatting and visualization
    - Interactive prediction mode
    - Memory-efficient processing
    """
    
    def __init__(self, model_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize AMR predictor.
        
        Args:
            model_path: Path to trained model
            logger: Optional logger instance
        """
        self.model_path = model_path
        self.logger = logger or setup_logger(__name__)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model and tokenizer."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.tokenizer = ViT5Tokenizer(logger=self.logger)
            self.tokenizer.tokenizer = self.tokenizer.tokenizer.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("✅ Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, input_text: str, max_length: int = 512, 
                num_beams: int = 4, do_sample: bool = False,
                temperature: float = 1.0) -> str:
        """
        Generate AMR prediction for a single input.
        
        Args:
            input_text: Input Vietnamese sentence
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            
        Returns:
            Generated AMR string
        """
        # Tokenize input
        inputs = self.tokenizer.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            if do_sample:
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=True,
                    temperature=temperature,
                    early_stopping=True
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )
        
        # Decode prediction
        prediction = self.tokenizer.decode_tokens(outputs[0], skip_special_tokens=True)
        return prediction.strip()
    
    def predict_batch(self, input_texts: List[str], batch_size: int = 8,
                     max_length: int = 512, num_beams: int = 4) -> List[str]:
        """
        Generate AMR predictions for a batch of inputs.
        
        Args:
            input_texts: List of input Vietnamese sentences
            batch_size: Batch size for processing
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            
        Returns:
            List of generated AMR strings
        """
        predictions = []
        
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode predictions
            for output in outputs:
                prediction = self.tokenizer.decode_tokens(output, skip_special_tokens=True)
                predictions.append(prediction.strip())
        
        return predictions
    
    def format_amr_penman(self, raw_amr: str) -> str:
        """
        Format AMR string in Penman notation with proper indentation.
        
        Args:
            raw_amr: Raw AMR string
            
        Returns:
            Formatted AMR string with proper indentation
        """
        if not raw_amr.strip():
            return raw_amr
        
        lines = []
        indent_level = 0
        indent_str = "   "  # 3 spaces for indentation
        
        # Split by lines and process each
        for line in raw_amr.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Count opening and closing parentheses
            open_parens = line.count('(')
            close_parens = line.count(')')
            
            # Adjust indentation for closing parentheses at the beginning
            if line.startswith(')'):
                indent_level = max(0, indent_level - close_parens)
            
            # Add indented line
            lines.append(indent_str * indent_level + line)
            
            # Adjust indentation for next line
            indent_level += open_parens - close_parens
            indent_level = max(0, indent_level)  # Ensure non-negative
        
        return '\n'.join(lines)
    
    def predict_and_format(self, input_text: str, **kwargs) -> Dict[str, str]:
        """
        Predict AMR and return both raw and formatted versions.
        
        Args:
            input_text: Input Vietnamese sentence
            **kwargs: Additional arguments for prediction
            
        Returns:
            Dictionary with 'raw' and 'formatted' AMR
        """
        raw_amr = self.predict(input_text, **kwargs)
        formatted_amr = self.format_amr_penman(raw_amr)
        
        return {
            'input': input_text,
            'raw': raw_amr,
            'formatted': formatted_amr
        }
    
    def interactive_mode(self) -> None:
        """
        Start interactive prediction mode.
        
        Allows users to input sentences and get AMR predictions in real-time.
        """
        self.logger.info("🚀 Starting interactive AMR prediction mode")
        self.logger.info("Enter Vietnamese sentences to get AMR predictions")
        self.logger.info("Type 'quit' or 'exit' to stop")
        
        print("\n" + "="*60)
        print("🤖 AMR Interactive Predictor")
        print("="*60)
        print("Enter Vietnamese sentences to get AMR predictions")
        print("Commands: 'quit', 'exit', 'help'")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("📝 Enter sentence: ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif not user_input:
                    print("⚠️  Please enter a sentence or 'quit' to exit")
                    continue
                
                # Generate prediction
                print("🔄 Generating AMR...")
                result = self.predict_and_format(user_input)
                
                # Display results
                print("\n" + "-"*50)
                print(f"📥 Input: {result['input']}")
                print(f"📤 AMR Output:")
                print(result['formatted'])
                print("-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.logger.error(f"Prediction error: {e}")
    
    def _show_help(self) -> None:
        """Show help information for interactive mode."""
        help_text = """
📖 Help - AMR Interactive Predictor

Commands:
  - Type any Vietnamese sentence to get AMR prediction
  - 'quit' or 'exit': Exit the program
  - 'help': Show this help message

Examples:
  - "Tôi yêu Việt Nam"
  - "Hôm nay trời đẹp"
  - "Cô ấy đang học tiếng Anh"

The system will generate Abstract Meaning Representation (AMR) 
for your input sentence in Penman notation format.
        """
        print(help_text)
    
    def predict_from_file(self, input_file: str, output_file: str,
                         batch_size: int = 8) -> None:
        """
        Predict AMR for sentences from a file.
        
        Args:
            input_file: Path to input file (one sentence per line)
            output_file: Path to output file
            batch_size: Batch size for processing
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing file: {input_file}")
        
        # Read input sentences
        with open(input_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        self.logger.info(f"Found {len(sentences)} sentences to process")
        
        # Generate predictions
        predictions = self.predict_batch(sentences, batch_size=batch_size)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence, prediction in zip(sentences, predictions):
                f.write(f"Input: {sentence}\n")
                f.write(f"AMR: {prediction}\n")
                f.write("-" * 50 + "\n")
        
        self.logger.info(f"✅ Results saved to: {output_file}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "vocab_size": self.tokenizer.get_vocab_size(),
            "max_length": self.tokenizer.max_length,
            "special_tokens": self.tokenizer.get_special_tokens()
        }
