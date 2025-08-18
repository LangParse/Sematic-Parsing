from typing import Dict
import evaluate
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EvalPrediction,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


class AMRTrainer:
    def __init__(
        self,
        dataset,
        model_name: str = "VietAI/vit5-base",
        save_dir: str = "./results",
        use_wandb: bool = False,
        run_name: str = "amr-train",
    ):
        """
        AMR Trainer class for seq2seq training with HuggingFace Trainer

        Args:
            model_name (str): Pretrained model name (e.g. "vinai/phobert-base").
            dataset (DatasetDict): Tokenized dataset with "train" and "validation".
            save_dir (str): Directory to save results and checkpoints.
            use_wandb (bool): Whether to log metrics to Weights & Biases.
            run_name (str): Run name for wandb logging.
        """

        self.model_name = model_name
        self.dataset = dataset
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.run_name = run_name

        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Initialize wandb
        if self.use_wandb:
            wandb.login(key="449ae55008e2e327116d3d500dfda77cdf77ce70")
            wandb.init(project="amr_parser", name=self.run_name)

    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        preds, labels = eval_pred

        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in labels (ignore_index) and decode
        decoded_labels = [[id for id in label if id != -100] for label in labels]
        decoded_labels = self.tokenizer.batch_decode(
            decoded_labels, skip_special_tokens=True
        )

        # Compute BLEU
        bleu = evaluate.load("bleu")
        bleu_score = (
            bleu.compute(predictions=decoded_preds, references=decoded_labels) or {}
        )

        # Optionally compute ROUGE as well
        rouge = evaluate.load("rouge")
        rouge_score = (
            rouge.compute(predictions=decoded_preds, references=decoded_labels) or {}
        )

        metrics = {
            "bleu": bleu_score.get("bleu", 0.0),
            "rouge1": rouge_score.get("rouge1", 0.0),
            "rougeL": rouge_score.get("rougeL", 0.0),
        }
        return metrics

    def train(self):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        training_args = TrainingArguments(
            output_dir=self.save_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            save_total_limit=3,
            logging_dir=f"{self.save_dir}/logs",
            logging_steps=100,
            logging_strategy="steps",
            report_to=["wandb"] if self.use_wandb else ["none"],
            run_name=self.run_name,
            fp16=True,
            # weight_decay=0.01,
            # load_best_model_at_end=True,
            # metric_for_best_model="eval_loss",
            # greater_is_better=False,
            # gradient_accumulation_steps=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            # data_collator=data_collator,
            # compute_metrics=self._compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        trainer.save_model(f"{self.save_dir}/amr_parser")
        self.tokenizer.save_pretrained(f"{self.save_dir}/amr_parser")

        if self.use_wandb:
            wandb.finish()

        return trainer
