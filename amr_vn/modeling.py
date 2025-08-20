
from transformers import T5ForConditionalGeneration

def load_model(model_name: str):
    return T5ForConditionalGeneration.from_pretrained(model_name)
