#!/usr/bin/env python3
"""
Gradio App for Vietnamese AMR Semantic Parsing
==============================================

A web interface for AMR semantic parsing using Gradio.
"""

import gradio as gr
import sys
import os
from pathlib import Path
import logging
import traceback

# Add src to path
sys.path.insert(0, 'src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AMRGradioApp:
    """Gradio app for AMR semantic parsing."""
    
    def __init__(self, model_path=None, hf_model_name=None):
        """Initialize the app."""
        self.model_path = model_path
        self.hf_model_name = hf_model_name
        self.predictor = None
        self.load_model()
    
    def load_model(self):
        """Load the AMR model."""
        try:
            if self.hf_model_name:
                logger.info(f"Loading model from Hugging Face: {self.hf_model_name}")
                self.predictor = self._load_hf_model()
            elif self.model_path:
                logger.info(f"Loading local model: {self.model_path}")
                self.predictor = self._load_local_model()
            else:
                logger.warning("No model specified, using mock predictor")
                self.predictor = self._create_mock_predictor()
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Using mock predictor instead")
            self.predictor = self._create_mock_predictor()
    
    def _load_hf_model(self):
        """Load model from Hugging Face."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_model_name)
            
            def predict(text):
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                amr = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return amr
            
            return predict
            
        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")
            raise
    
    def _load_local_model(self):
        """Load local model."""
        try:
            from src.inference import AMRPredictor
            
            predictor = AMRPredictor(self.model_path, logger=logger)
            return lambda text: predictor.predict(text)
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def _create_mock_predictor(self):
        """Create mock predictor for demo."""
        def mock_predict(text):
            # Simple mock AMR generation
            words = text.split()
            if len(words) == 0:
                return "(empty / text)"
            
            # Create a simple mock AMR structure
            main_concept = words[0].lower()
            amr_parts = [f"({main_concept[0]} / {main_concept}"]
            
            for i, word in enumerate(words[1:], 1):
                amr_parts.append(f"   :ARG{i} \"{word}\"")
            
            amr_parts.append(")")
            return "\n".join(amr_parts)
        
        return mock_predict
    
    def predict_amr(self, text):
        """Predict AMR for input text."""
        if not text or not text.strip():
            return "Vui lòng nhập câu tiếng Việt để phân tích AMR."
        
        try:
            # Clean input
            text = text.strip()
            
            # Predict AMR
            amr = self.predictor(text)
            
            # Format AMR for better display
            formatted_amr = self._format_amr(amr)
            
            return formatted_amr
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return f"Lỗi khi phân tích: {str(e)}"
    
    def _format_amr(self, amr):
        """Format AMR for better display."""
        if not amr:
            return "Không thể tạo AMR cho câu này."
        
        # Simple formatting - add line breaks after colons
        formatted = amr.replace(" :", "\n   :")
        return formatted
    
    def create_interface(self):
        """Create Gradio interface."""
        
        # Custom CSS for styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .input-text {
            font-size: 16px !important;
        }
        .output-text {
            font-family: 'Courier New', monospace !important;
            font-size: 14px !important;
            background-color: #f8f9fa !important;
        }
        """
        
        # Create interface
        with gr.Blocks(css=css, title="Vietnamese AMR Parser") as interface:
            
            # Header
            gr.Markdown("""
            # 🇻🇳 Vietnamese AMR Semantic Parsing
            
            Nhập câu tiếng Việt để phân tích cấu trúc ngữ nghĩa AMR (Abstract Meaning Representation).
            
            **Ví dụ:** "Tôi yêu Việt Nam", "Cô ấy đang học tiếng Anh", "Hôm nay trời đẹp"
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input
                    input_text = gr.Textbox(
                        label="Nhập câu tiếng Việt",
                        placeholder="Ví dụ: Tôi yêu Việt Nam",
                        lines=3,
                        elem_classes=["input-text"]
                    )
                    
                    # Buttons
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column(scale=1):
                    # Output
                    output_text = gr.Textbox(
                        label="AMR Output",
                        lines=10,
                        elem_classes=["output-text"],
                        interactive=False
                    )
            
            # Examples
            gr.Examples(
                examples=[
                    ["Tôi yêu Việt Nam"],
                    ["Cô ấy đang học tiếng Anh"],
                    ["Hôm nay trời đẹp"],
                    ["Bác sĩ khám bệnh cho bệnh nhân"],
                    ["Sinh viên đang thi cuối kỳ"],
                    ["Mẹ tôi nấu cơm rất ngon"]
                ],
                inputs=input_text,
                outputs=output_text,
                fn=self.predict_amr,
                cache_examples=False
            )
            
            # Footer
            gr.Markdown("""
            ---
            **Powered by:** VietAI/vit5-base | **Framework:** Transformers + Gradio
            
            **Lưu ý:** Đây là mô hình demo. Chất lượng AMR phụ thuộc vào dữ liệu training.
            """)
            
            # Event handlers
            submit_btn.click(
                fn=self.predict_amr,
                inputs=input_text,
                outputs=output_text
            )
            
            input_text.submit(
                fn=self.predict_amr,
                inputs=input_text,
                outputs=output_text
            )
            
            clear_btn.click(
                fn=lambda: ("", ""),
                outputs=[input_text, output_text]
            )
        
        return interface

def main():
    """Main function to run the Gradio app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese AMR Gradio App")
    parser.add_argument("--model-path", type=str, help="Path to local trained model")
    parser.add_argument("--hf-model", type=str, help="Hugging Face model name")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the app")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create app
    app = AMRGradioApp(
        model_path=args.model_path,
        hf_model_name=args.hf_model
    )
    
    # Create interface
    interface = app.create_interface()
    
    # Launch
    print(f"🚀 Starting Vietnamese AMR Gradio App...")
    print(f"   Model: {args.hf_model or args.model_path or 'Mock'}")
    print(f"   URL: http://{args.host}:{args.port}")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
