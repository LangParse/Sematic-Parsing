from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr


class AMRDemo:
    def __init__(
        self,
        model_dir: str = "./models/v1/amr_parser",
        max_input_len: int = 256,
        max_output_len: int = 512,
    ):
        """Khởi tạo Gradio demo cho AMR Parser"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)

        # Load tokenizer & model đã fine-tuned
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_dir), local_files_only=True
        ).to(self.device)

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def predict(self, sentence: str) -> str:
        """Sinh AMR từ 1 câu input"""
        if not sentence.strip():
            return "⚠️ Vui lòng nhập một câu."

        # Chuẩn bị input
        inputs = self.tokenizer(
            ["semantic parse: " + sentence],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_input_len,
        ).to(self.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_len,
            num_beams=6,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded

    def launch(
        self, server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False
    ):
        """Khởi chạy Gradio UI"""
        demo = gr.Interface(
            fn=self.predict,
            inputs=gr.Textbox(
                lines=5,
                placeholder="Nhập một câu tiếng Việt...",
                label="Input Sentence",
            ),
            outputs=gr.Textbox(
                lines=12, placeholder="Kết quả AMR graph...", label="Predicted AMR"
            ),
            title="Vietnamese AMR Parser Demo",
            description="Nhập một câu tiếng Việt và nhận về đồ thị AMR (PENMAN format).",
        )
        demo.launch(server_name=server_name, server_port=server_port, share=share)
