import json

from nltk.translate.bleu_score import sentence_bleu


def evaluate_model(model, tokenizer, json_path, max_samples=100):
    """
    Evaluate the model using BLEU score on a sample of the dataset.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    total_score = 0
    device = next(model.parameters()).device
    for sample in samples[:max_samples]:
        input_ids = tokenizer(sample["input"], return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids)[0]
        prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
        reference = sample["output"].split()
        hypothesis = prediction.split()
        bleu_score = sentence_bleu([reference], hypothesis)
        if isinstance(bleu_score, list):
            bleu_score = float(bleu_score[0]) if bleu_score else 0.0
        total_score += bleu_score
    avg_bleu = round(total_score / max_samples, 4)
    print("BLEU score:", avg_bleu)
    return avg_bleu
