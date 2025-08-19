import logging
from typing import Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch
import penman
from amrlib.evaluate.smatch_enhanced import compute_smatch


class AMREvaluator:
    """
    Class hỗ trợ đánh giá mô hình AMR đã fine-tuned.
    Bao gồm các chức năng:
        - Tokenize dataset để evaluate
        - Tính Smatch F1/Precision/Recall
        - Sinh dự đoán AMR trên tập test
    """

    def __init__(
        self,
        model_dir: str,
        dataset: Dict[str, Dataset],
        max_input_len: int = 256,
        max_output_len: int = 512,
    ):
        """
        Args:
            model_dir (str): Thư mục chứa model đã fine-tuned.
            dataset (Dict[str, Dataset]): DatasetDict gồm train/validation/test.
            max_input_len (int): Độ dài tối đa cho input tokens.
            max_output_len (int): Độ dài tối đa cho output tokens.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.dataset = dataset
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def _preprocess_function(self, examples):
        """
        Tokenize input và output cho model.

        Args:
            examples (dict): Batch chứa "input" (câu gốc) và "output" (AMR graph).

        Returns:
            dict: Batch chứa input_ids, attention_mask, labels.
        """
        model_inputs = self.tokenizer(
            examples["input"],
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["output"],
                max_length=self.max_output_len,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _compute_metrics(self, eval_preds):
        """
        Hàm tính Smatch F1/Precision/Recall.

        Args:
            eval_preds (tuple): (predictions, labels) từ Seq2SeqTrainer.

        Returns:
            dict: {"smatch_f1": ..., "precision": ..., "recall": ...}
        """
        preds, labels = eval_preds
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        valid_preds, valid_labels = [], []
        for p, l in zip(decoded_preds, decoded_labels):
            try:
                penman.decode(p)
                penman.decode(l)
                valid_preds.append(p)
                valid_labels.append(l)
            except Exception:
                continue

        if not valid_preds:
            return {"smatch_f1": 0.0, "precision": 0.0, "recall": 0.0}

        pred, rec, f1 = compute_smatch(valid_preds, valid_labels)

        return {
            "smatch_f1": f1,
            "precision": pred,
            "recall": rec,
        }

    def evaluate(self, split: str = "validation") -> Dict[str, Any]:
        """
        Đánh giá model trên split (validation hoặc test nếu có gold AMR).

        Args:
            split (str): Tên split trong dataset.

        Returns:
            dict: Kết quả đánh giá gồm smatch_f1, precision, recall.
        """
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not có trong dataset")

        eval_ds = self.dataset[split]
        self.logger.info(f"Evaluating on split: {eval_ds}")

        if "output" not in eval_ds.column_names:
            self.logger.warning(f"Split '{split}' không có gold AMR → bỏ qua evaluate")
            return {}

        # tokenize dataset
        tokenized_ds = eval_ds.map(
            self._preprocess_function,
            batched=True,
            remove_columns=eval_ds.column_names,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir="./reports",
            per_device_eval_batch_size=4,
            predict_with_generate=True,
            generation_max_length=self.max_output_len,
            generation_num_beams=6,
            remove_unused_columns=False,
            logging_dir="./reports/logs",
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            compute_metrics=self._compute_metrics,
        )

        self.logger.info(f"Evaluating model on split: {split}")
        results = trainer.evaluate(eval_dataset=tokenized_ds)  # pyright: ignore
        self.logger.info(f"Evaluation results: {results}")
        return results

    def generate_predictions(self, split: str = "test", num_samples: int = 5):
        """
        Sinh AMR graph cho tập test (không cần gold AMR).

        Args:
            split (str): Tên split trong dataset.
            num_samples (int): Số câu muốn sinh kết quả.

        Returns:
            list[tuple]: Danh sách (câu gốc, AMR dự đoán).
        """
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not có trong dataset")

        test_ds = self.dataset[split]
        inputs = test_ds["input"][:num_samples]

        model_inputs = self.tokenizer(
            ["semantic parse: " + s for s in inputs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_len,
        ).to(self.device)

        outputs = self.model.generate(
            **model_inputs,
            max_length=self.max_output_len,
            num_beams=6,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return list(zip(inputs, decoded))
