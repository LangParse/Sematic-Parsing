import os
import torch
from typing import Dict
import evaluate
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.amr_tokenizer import AMRTokenizer


class AMRTrainer:
    def __init__(
        self,
        dataset,
        model_name: str = "VietAI/vit5-base",
        save_dir: str = "./models",
        use_wandb: bool = False,
        run_name: str = "amr-train",
        max_input_len: int = 128,
        max_output_len: int = 256,
    ):
        """
        AMR Trainer class for seq2seq training with HuggingFace Trainer

        Args:
            dataset (DatasetDict): Raw dataset with "train"/"validation" (and optional "test").
            model_name (str): Pretrained model (e.g. "VietAI/vit5-base").
            save_dir (str): Directory to save results and checkpoints.
            use_wandb (bool): Whether to log metrics to Weights & Biases.
            run_name (str): Run name for wandb logging.
            max_input_len (int): Max length for input sentences.
            max_output_len (int): Max length for AMR graphs.
        """

        self.model_name = model_name
        self.dataset = dataset
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.run_name = run_name

        # Load tokenizer wrapper
        self.amr_tokenizer = AMRTokenizer(
            model_name=model_name,
            max_length_input=max_input_len,
            max_length_output=max_output_len,
        )

        # Tokenize dataset before training
        self.tokenized_dataset = self.amr_tokenizer.tokenize_dataset(self.dataset)

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        # Init wandb if needed
        if self.use_wandb:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(project="amr_parser", name=self.run_name)

    @staticmethod
    def _compute_metrics(eval_preds) -> Dict[str, float]:
        """Compute BLEU & ROUGE after evaluation."""
        preds, labels = eval_preds
        tokenizer = AMRTokenizer().tokenizer

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = [[id for id in label if id != -100] for label in labels]
        decoded_labels = tokenizer.batch_decode(
            decoded_labels, skip_special_tokens=True
        )

        # BLEU
        bleu = evaluate.load("bleu")
        bleu_score = (
            bleu.compute(predictions=decoded_preds, references=decoded_labels) or {}
        )

        # ROUGE
        rouge = evaluate.load("rouge")
        rouge_score = (
            rouge.compute(predictions=decoded_preds, references=decoded_labels) or {}
        )

        return {
            "bleu": bleu_score.get("bleu", 0.0),
            "rouge1": rouge_score.get("rouge1", 0.0),
            "rougeL": rouge_score.get("rougeL", 0.0),
        }

    def train(self, num_train_epochs: int = 5, batch_size: int = 8):
        """Run training loop."""
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.amr_tokenizer.tokenizer,
            model=self.model,
            padding=True,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.save_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            logging_dir=f"{self.save_dir}/logs",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            report_to=["wandb"] if self.use_wandb else [],
            run_name=self.run_name,
            fp16=True,
            remove_unused_columns=False,
            prediction_loss_only=False,  # vẫn tính metrics ở cuối
            eval_strategy="epoch",  # đánh giá mỗi epoch (ổn hơn so với quá thường xuyên)
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset.get("train"),
            eval_dataset=self.tokenized_dataset.get("validation"),  # pyright: ignore
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()

        # Save final model
        model_ver = os.getenv("MODEL_VERSION", "base")
        save_path = f"{self.save_dir}/{model_ver}/amr_parser"
        trainer.save_model(save_path)
        self.amr_tokenizer.tokenizer.save_pretrained(save_path)

        if self.use_wandb:
            wandb.finish()

        return trainer
