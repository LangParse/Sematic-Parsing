import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Optional
import torch
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model  # Thêm PEFT LoRA
from amrlib.evaluate.smatch_enhanced import compute_smatch
import optuna  # Thêm cho tune
import penman

from src.amr_tokenizer import AMRTokenizer


class AMRTrainer:
    def __init__(
        self,
        dataset,
        model_name: str = "vietai/vit5-base",
        save_dir: str = "./models/v1",
        use_wandb: bool = False,
        run_name: str = "amr-train",
        max_input_len: int = 256,
        max_output_len: int = 512,
        eval_limit: Optional[int] = None,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.run_name = run_name
        self.eval_limit = eval_limit

        # Load tokenizer
        self.amr_tokenizer = AMRTokenizer(
            model_name=model_name,
            max_length_input=max_input_len,
            max_length_output=max_output_len,
        )

        # Tokenize dataset
        self.tokenized_dataset = self.amr_tokenizer.tokenize_dataset(self.dataset)

        # Load model with LoRA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.gradient_checkpointing_enable()

        # Freeze encoder layers để giảm overfit
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Áp dụng LoRA
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q", "v"],  # Modules to adapt
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)

        # Init wandb
        if self.use_wandb:
            wandb.login(key="449ae55008e2e327116d3d500dfda77cdf77ce70")
            wandb.init(project="amr_parser", name=self.run_name)

    def _compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Compute Smatch after evaluation, with error handling for ill-formatted AMR."""
        preds, labels = eval_preds
        tokenizer = self.amr_tokenizer.tokenizer

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels (ignore -100)
        decoded_labels = [[id for id in label if id != -100] for label in labels]
        decoded_labels = tokenizer.batch_decode(
            decoded_labels, skip_special_tokens=True
        )

        if self.eval_limit is not None:
            decoded_preds = decoded_preds[: self.eval_limit]
            decoded_labels = decoded_labels[: self.eval_limit]

        # Filter valid AMR pairs
        valid_preds = []
        valid_labels = []
        for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            try:
                # Validate format với penman
                penman.decode(pred)  # Check pred
                penman.decode(label)  # Check label (dù gold ok, nhưng double-check)
                valid_preds.append(pred)
                valid_labels.append(label)
            except (penman.DecodeError, Exception) as e:
                error_msg = f"Skipping ill-formatted AMR at index {idx}: Pred='{pred[:100]}...', Label='{label[:100]}...'. Error: {e}"
                print(error_msg)  # In console
                logging.error(error_msg)  # Ghi file amr_errors.log

        if not valid_preds:
            print("All AMRs invalid - Returning zero scores. Check model generation.")
            return {"smatch_f1": 0.0, "smatch_precision": 0.0, "smatch_recall": 0.0}

        # Tính Smatch (amrlib wrap smatch)
        prec, rec, f1 = compute_smatch(
            valid_preds, valid_labels
        )  # Trả về dict với f1, precision, recall

        return {
            "smatch_f1": f1,
            "smatch_precision": prec,
            "smatch_recall": rec,
        }

    def train(
        self,
        num_train_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
    ):
        """Run training loop."""
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.amr_tokenizer.tokenizer,
            model=self.model,
            padding=True,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.save_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=2,
            num_train_epochs=num_train_epochs,
            logging_dir=f"{self.save_dir}/logs",
            logging_steps=100,
            save_total_limit=2,
            report_to=["wandb"] if self.use_wandb else [],
            run_name=self.run_name,
            fp16=True,
            remove_unused_columns=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="smatch_f1",  # Dùng Smatch làm best metric
            greater_is_better=True,
            predict_with_generate=True,
            eval_accumulation_steps=1,
            learning_rate=learning_rate,  # Set explicit
            lr_scheduler_type="linear",  # Thêm scheduler
            warmup_steps=100,  # Warmup
            generation_num_beams=6,  # Thêm để generate AMR tốt hơn, giảm ill-formatted
            generation_max_length=self.amr_tokenizer.max_length_output,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset.get("train"),
            eval_dataset=self.tokenized_dataset.get("validation"),  # pyright: ignore
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3)
            ],  # Early stop nếu val không cải thiện
        )

        trainer.train()

        # Save
        save_path = f"{self.save_dir}/amr_parser"
        trainer.save_model(save_path)
        self.amr_tokenizer.tokenizer.save_pretrained(save_path)

        if self.use_wandb:
            wandb.finish()

        return trainer

    def tune_hyperparams(self, n_trials: int = 20):
        """Tune siêu tham số với Optuna."""

        def objective(trial):
            params = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
                "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            }
            trainer = self.train(**params)
            eval_results = trainer.evaluate()
            return eval_results["eval_smatch_f1"]  # Maximize Smatch F1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print("Best params:", study.best_params)
        return study.best_params
