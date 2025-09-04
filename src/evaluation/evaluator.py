"""
AMR Evaluator Module
===================

This module handles model evaluation with comprehensive metrics and visualization.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration

from ..tokenization import ViT5Tokenizer
from ..utils import setup_logger


@dataclass
class EvaluationMetrics:
    """Data class for storing evaluation metrics."""

    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0
    meteor: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    exact_match: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "bleu_1": self.bleu_1,
            "bleu_2": self.bleu_2,
            "bleu_3": self.bleu_3,
            "bleu_4": self.bleu_4,
            "meteor": self.meteor,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "rouge_l": self.rouge_l,
            "exact_match": self.exact_match,
        }


class AMREvaluator:
    """
    Comprehensive evaluator for AMR models.

    This class provides:
    - Multiple evaluation metrics (BLEU, METEOR, ROUGE)
    - Batch evaluation capabilities
    - Visualization of results
    - Detailed error analysis
    """

    def __init__(self, model_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize AMR evaluator.

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

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        self._load_model()

    def _load_model(self) -> None:
        """Load trained model and tokenizer."""
        self.logger.info(f"Loading model from: {self.model_path}")

        try:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.tokenizer = ViT5Tokenizer(logger=self.logger)
            self.tokenizer.tokenizer = self.tokenizer.tokenizer.from_pretrained(
                self.model_path
            )

            self.model.to(self.device)
            self.model.eval()

            self.logger.info("✅ Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict_single(self, input_text: str, max_length: int = 512) -> str:
        """
        Generate AMR prediction for a single input.

        Args:
            input_text: Input sentence
            max_length: Maximum generation length

        Returns:
            Generated AMR string
        """
        # Tokenize input
        inputs = self.tokenizer.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
            )

        # Decode prediction
        prediction = self.tokenizer.decode_tokens(outputs[0], skip_special_tokens=True)
        return prediction

    def predict_batch(
        self, input_texts: List[str], batch_size: int = 8, max_length: int = 512
    ) -> List[str]:
        """
        Generate AMR predictions for a batch of inputs.

        Args:
            input_texts: List of input sentences
            batch_size: Batch size for processing
            max_length: Maximum generation length

        Returns:
            List of generated AMR strings
        """
        predictions = []

        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            # Generate predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                )

            # Decode predictions
            for output in outputs:
                prediction = self.tokenizer.decode_tokens(
                    output, skip_special_tokens=True
                )
                predictions.append(prediction)

        return predictions

    def calculate_bleu_scores(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate BLEU scores.

        Args:
            predictions: List of predicted AMR strings
            references: List of reference AMR strings

        Returns:
            Dictionary with BLEU scores
        """
        # Tokenize for BLEU calculation
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]

        # Calculate individual BLEU scores
        bleu_1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))

        return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}

    def calculate_rouge_scores(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            predictions: List of predicted AMR strings
            references: List of reference AMR strings

        Returns:
            Dictionary with ROUGE scores
        """
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_1_scores.append(scores["rouge1"].fmeasure)
            rouge_2_scores.append(scores["rouge2"].fmeasure)
            rouge_l_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge_1": sum(rouge_1_scores) / len(rouge_1_scores),
            "rouge_2": sum(rouge_2_scores) / len(rouge_2_scores),
            "rouge_l": sum(rouge_l_scores) / len(rouge_l_scores),
        }

    def calculate_exact_match(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """
        Calculate exact match accuracy.

        Args:
            predictions: List of predicted AMR strings
            references: List of reference AMR strings

        Returns:
            Exact match accuracy
        """
        exact_matches = sum(
            1
            for pred, ref in zip(predictions, references)
            if pred.strip() == ref.strip()
        )
        return exact_matches / len(predictions)

    def evaluate_dataset(
        self, test_data: List[Dict[str, str]], output_dir: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        Evaluate model on a test dataset.

        Args:
            test_data: List of test samples with 'input' and 'output' keys
            output_dir: Optional directory to save results

        Returns:
            EvaluationMetrics object
        """
        self.logger.info(f"Evaluating on {len(test_data)} samples...")

        # Extract inputs and references
        inputs = [sample["input"] for sample in test_data]
        references = [sample["output"] for sample in test_data]

        # Generate predictions
        predictions = self.predict_batch(inputs)

        # Calculate metrics
        bleu_scores = self.calculate_bleu_scores(predictions, references)
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        exact_match = self.calculate_exact_match(predictions, references)

        # Create metrics object
        metrics = EvaluationMetrics(
            bleu_1=bleu_scores["bleu_1"],
            bleu_2=bleu_scores["bleu_2"],
            bleu_3=bleu_scores["bleu_3"],
            bleu_4=bleu_scores["bleu_4"],
            rouge_1=rouge_scores["rouge_1"],
            rouge_2=rouge_scores["rouge_2"],
            rouge_l=rouge_scores["rouge_l"],
            exact_match=exact_match,
        )

        # Log results
        self.logger.info("Evaluation Results:")
        for key, value in metrics.to_dict().items():
            self.logger.info(f"{key}: {value:.4f}")

        # Save results if output directory specified
        if output_dir:
            self._save_evaluation_results(
                metrics, predictions, references, inputs, output_dir
            )

        return metrics

    def _save_evaluation_results(
        self,
        metrics: EvaluationMetrics,
        predictions: List[str],
        references: List[str],
        inputs: List[str],
        output_dir: str,
    ) -> None:
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = output_path / "evaluation_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        # Save detailed results
        results_file = output_path / "detailed_results.jsonl"
        with open(results_file, "w", encoding="utf-8") as f:
            for inp, pred, ref in zip(inputs, predictions, references):
                result = {
                    "input": inp,
                    "prediction": pred,
                    "reference": ref,
                    "exact_match": pred.strip() == ref.strip(),
                }
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")

        self.logger.info(f"Evaluation results saved to: {output_dir}")

    def create_evaluation_report(
        self, metrics: EvaluationMetrics, output_dir: str
    ) -> None:
        """Create comprehensive evaluation report with visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create metrics visualization
        self._plot_metrics(metrics, str(output_path / "metrics_plot.png"))

        # Create evaluation report
        report_file = output_path / "evaluation_report.md"
        self._generate_markdown_report(metrics, str(report_file))

        self.logger.info(f"Evaluation report created: {report_file}")

    def _plot_metrics(self, metrics: EvaluationMetrics, output_file: str) -> None:
        """Create visualization of evaluation metrics."""
        plt.figure(figsize=(12, 8))

        # Prepare data
        metric_names = list(metrics.to_dict().keys())
        metric_values = list(metrics.to_dict().values())

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # BLEU scores
        bleu_names = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
        bleu_values = [metrics.bleu_1, metrics.bleu_2, metrics.bleu_3, metrics.bleu_4]
        ax1.bar(bleu_names, bleu_values, color="skyblue")
        ax1.set_title("BLEU Scores")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1)

        # ROUGE scores
        rouge_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
        rouge_values = [metrics.rouge_1, metrics.rouge_2, metrics.rouge_l]
        ax2.bar(rouge_names, rouge_values, color="lightcoral")
        ax2.set_title("ROUGE Scores")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1)

        # Overall comparison
        all_metrics = ["BLEU-4", "ROUGE-L", "Exact Match"]
        all_values = [metrics.bleu_4, metrics.rouge_l, metrics.exact_match]
        ax3.bar(all_metrics, all_values, color="lightgreen")
        ax3.set_title("Key Metrics Comparison")
        ax3.set_ylabel("Score")
        ax3.set_ylim(0, 1)

        # Radar chart for all metrics
        angles = [n / len(metric_names) * 2 * 3.14159 for n in range(len(metric_names))]
        angles += angles[:1]  # Complete the circle

        values = metric_values + metric_values[:1]  # Complete the circle

        ax4 = plt.subplot(224, projection="polar")
        ax4.plot(angles, values, "o-", linewidth=2, color="purple")
        ax4.fill(angles, values, alpha=0.25, color="purple")
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metric_names)
        ax4.set_ylim(0, 1)
        ax4.set_title("All Metrics Overview")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_markdown_report(
        self, metrics: EvaluationMetrics, output_file: str
    ) -> None:
        """Generate markdown evaluation report."""
        report_content = f"""# AMR Model Evaluation Report

## Overview
This report contains the evaluation results for the AMR semantic parsing model.

## Metrics Summary

### BLEU Scores
- **BLEU-1**: {metrics.bleu_1:.4f}
- **BLEU-2**: {metrics.bleu_2:.4f}
- **BLEU-3**: {metrics.bleu_3:.4f}
- **BLEU-4**: {metrics.bleu_4:.4f}

### ROUGE Scores
- **ROUGE-1**: {metrics.rouge_1:.4f}
- **ROUGE-2**: {metrics.rouge_2:.4f}
- **ROUGE-L**: {metrics.rouge_l:.4f}

### Other Metrics
- **Exact Match**: {metrics.exact_match:.4f}

## Analysis

### Performance Summary
The model achieved a BLEU-4 score of {metrics.bleu_4:.4f}, which indicates {"good" if metrics.bleu_4 > 0.3 else "moderate" if metrics.bleu_4 > 0.1 else "low"} performance in generating AMR representations.

The ROUGE-L score of {metrics.rouge_l:.4f} suggests {"strong" if metrics.rouge_l > 0.5 else "moderate" if metrics.rouge_l > 0.3 else "weak"} overlap with reference AMR structures.

The exact match accuracy of {metrics.exact_match:.4f} shows that {metrics.exact_match * 100:.1f}% of predictions exactly match the reference AMR.

### Recommendations
{"The model shows strong performance across all metrics." if metrics.bleu_4 > 0.4 and metrics.rouge_l > 0.5 else "Consider additional training or data augmentation to improve performance." if metrics.bleu_4 < 0.2 else "The model shows reasonable performance but could benefit from fine-tuning."}

## Visualization
See `metrics_plot.png` for visual representation of the evaluation metrics.

---
*Report generated automatically by AMR Evaluator*
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

    def compare_models(
        self,
        other_evaluator: "AMREvaluator",
        test_data: List[Dict[str, str]],
        output_dir: str,
    ) -> None:
        """Compare this model with another model."""
        self.logger.info("Comparing models...")

        # Evaluate both models
        metrics_1 = self.evaluate_dataset(test_data)
        metrics_2 = other_evaluator.evaluate_dataset(test_data)

        # Create comparison report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        comparison_data = {
            "Model 1": metrics_1.to_dict(),
            "Model 2": metrics_2.to_dict(),
        }

        # Save comparison
        comparison_file = output_path / "model_comparison.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)

        # Create comparison visualization
        self._plot_model_comparison(
            metrics_1, metrics_2, str(output_path / "comparison_plot.png")
        )

        self.logger.info(f"Model comparison saved to: {output_dir}")

    def _plot_model_comparison(
        self,
        metrics_1: EvaluationMetrics,
        metrics_2: EvaluationMetrics,
        output_file: str,
    ) -> None:
        """Create comparison plot between two models."""
        metric_names = list(metrics_1.to_dict().keys())
        values_1 = list(metrics_1.to_dict().values())
        values_2 = list(metrics_2.to_dict().values())

        x = range(len(metric_names))
        width = 0.35

        plt.figure(figsize=(15, 8))
        plt.bar([i - width / 2 for i in x], values_1, width, label="Model 1", alpha=0.8)
        plt.bar([i + width / 2 for i in x], values_2, width, label="Model 2", alpha=0.8)

        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Model Comparison")
        plt.xticks(x, metric_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
