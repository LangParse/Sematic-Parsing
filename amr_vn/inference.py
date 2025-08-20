
from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from .utils.penman import basic_penman_format

class AMRPredictor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        from transformers import T5ForConditionalGeneration
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

    def predict_one(self, sentence: str, max_length: int = 512) -> str:
        input_ids = self.tokenizer(sentence, return_tensors="pt").input_ids.to(self.model.device)
        output_ids = self.model.generate(input_ids, max_length=max_length)[0]
        out = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return basic_penman_format(out)

    def predict_file(self, sentences: List[str], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fo:
            for s in sentences:
                amr = self.predict_one(s)
                fo.write(f"Câu: {s}\nAMR dự đoán:\n{amr}\n" + "-"*60 + "\n")
