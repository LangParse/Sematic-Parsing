
from dataclasses import asdict
from typing import List, Dict
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from .config import TrainConfig

def build_dataset(samples: List[Dict[str, str]]) -> Dataset:
    return Dataset.from_list(samples)

def build_trainer(model, tokenizer, tokenized_ds: Dataset, cfg: TrainConfig, output_dir: str):
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        seed=cfg.seed,
        report_to=[],
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    return trainer
