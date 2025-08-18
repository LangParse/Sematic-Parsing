import json
import os

import wandb
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
)
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments


def train_model(
    json_path,
    model_output_dir,
    log_dir,
    model_name="VietAI/vit5-base",
    run_name="AMR_ViT5_T4GPU",
):
    """
    Train a T5-based model for AMR parsing.
    """
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        raise ValueError("WANDB_API_KEY not found in .env file.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    def preprocess(example):
        model_input = tokenizer(
            example["input"], padding="max_length", truncation=True, max_length=512
        )
        labels = tokenizer(
            example["output"], padding="max_length", truncation=True, max_length=512
        )
        model_input["labels"] = labels["input_ids"]
        return model_input

    tokenized_dataset = dataset.map(preprocess)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=10,
        logging_dir=log_dir,
        logging_steps=500,
        save_steps=200,
        save_total_limit=2,
        report_to=["wandb"],
        run_name=run_name,
        fp16=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    return model, tokenizer
