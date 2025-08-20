from __future__ import annotations
from typing import Optional
import gradio as gr

# local predictor (đã có sẵn)
from .inference import AMRPredictor
from .utils.penman import format_amr_penman

# thêm: hỗ trợ HF
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer


class GradioDemo:
    """
    Demo Gradio cho mô hình AMR.
    - Local: dùng AMRPredictor(model_path)
    - HF Hub: truyền hf_id để tải trực tiếp từ Hugging Face
    """

    def __init__(self, model_path: Optional[str] = None, hf_id: Optional[str] = None):
        self.hf = None
        if hf_id:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                tok = T5TokenizerFast.from_pretrained(hf_id)
            except Exception:
                tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
            mdl = T5ForConditionalGeneration.from_pretrained(hf_id).to(device).eval()
            self.hf = {"tokenizer": tok, "model": mdl, "device": device}
            self.model_path = hf_id
            self.predictor = None
        else:
            assert model_path, "model_path hoặc hf_id là bắt buộc"
            self.model_path = model_path
            self.predictor = AMRPredictor(model_path)

    # ---- inference ----
    def _predict_hf(
        self, text: str, max_new_tokens: int = 128, num_beams: int = 4
    ) -> str:
        tok = self.hf["tokenizer"]
        mdl = self.hf["model"]
        device = self.hf["device"]
        inputs = tok([text], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                num_beams=int(num_beams),
                length_penalty=0.6,
                early_stopping=True,
            )
        return tok.decode(out[0], skip_special_tokens=True)

    def predict(self, text: str, max_new_tokens: int = 128, num_beams: int = 4) -> str:
        if not text or not text.strip():
            return ""
        if self.hf:
            return self._predict_hf(
                text.strip(), max_new_tokens=max_new_tokens, num_beams=num_beams
            )
        return self.predictor.predict_one(text.strip())

    # wrapper để format ra UI
    def predict_formatted(self, text: str) -> str:
        raw = self.predict(text)  # dùng default beam/token như bạn set
        return format_amr_penman(raw) if raw else ""

    def launch(
        self, server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False
    ):
        """Khởi chạy Gradio UI"""
        demo = gr.Interface(
            fn=self.predict_formatted,
            inputs=gr.Textbox(
                lines=12,
                placeholder="Nhập một câu tiếng Việt...",
                label="Input Sentence",
            ),
            outputs=gr.Textbox(
                lines=12,
                placeholder="Kết quả AMR graph...",
                label="Predicted AMR",
            ),
            title="Vietnamese AMR Parser Demo",
            description="Nhập một câu tiếng Việt và nhận về đồ thị AMR (PENMAN format).",
        )
        demo.launch(server_name=server_name, server_port=server_port, share=share)
