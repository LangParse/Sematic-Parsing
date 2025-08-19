import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
from typing import List, Optional


class AMRPredictor:
    def __init__(
        self, model_dir: str = "./models/v1/amr_parser", max_length: int = 512
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        self.max_length = max_length

    def predict(self, sentences: List[str]) -> List[str]:
        """Nhận list câu đầu vào, trả về list AMR graph (string)."""
        inputs = ["semantic parse: " + s for s in sentences]
        tokenized = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **tokenized,
            max_length=self.max_length,
            num_beams=6,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    def predict_file(self, test_path: str, output_path: Optional[str] = None):
        """Dự đoán AMR cho file test (mỗi dòng 1 câu).
        Nếu có output_path thì lưu ra file giống format train_split.txt
        """
        with open(test_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        preds = self.predict(sentences)

        if output_path:
            self._save_to_file(sentences, preds, output_path)

        return list(zip(sentences, preds))

    def _save_to_file(self, sentences: List[str], preds: List[str], output_path: str):
        """Lưu kết quả ra file text theo format AMR chuẩn"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for sent, amr in zip(sentences, preds):
                f.write(f"#::snt {sent}\n")
                # format lại AMR: tách dòng sau mỗi ')'
                formatted = self._format_amr(amr)
                f.write(formatted.strip() + "\n\n")

        print(f"Predictions saved to {output_path}")

    def _format_amr(self, amr_str: str) -> str:
        """Thử format AMR cho dễ đọc: xuống dòng sau mỗi ')', thụt đầu dòng"""
        lines = []
        indent = 0
        token = ""
        for ch in amr_str:
            if ch == "(":
                if token.strip():
                    lines.append(" " * indent + token.strip())
                token = "("
                indent += 4
            elif ch == ")":
                if token.strip():
                    lines.append(" " * indent + token.strip())
                token = ")"
                indent -= 4
                lines.append(" " * indent + ")")
                token = ""
            elif ch == "\n":
                if token.strip():
                    lines.append(" " * indent + token.strip())
                token = ""
            else:
                token += ch

        if token.strip():
            lines.append(" " * indent + token.strip())

        return "\n".join(lines)
