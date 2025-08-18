import json

from nltk.translate.bleu_score import sentence_bleu


def evaluate_model(model, tokenizer, json_path, max_samples=100):
    """
    Evaluate a sequence-to-sequence model using BLEU score on a sample of the dataset.
    Args:
        model: The PyTorch model to evaluate.
        tokenizer: The tokenizer for encoding/decoding text.
        json_path (str): Path to the JSON file with input/output pairs.
        max_samples (int): Number of samples to evaluate.
    Returns:
        float: The average BLEU score over the evaluated samples.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not samples:
        print("No samples found in the dataset.")
        return 0.0
    total_score = 0.0
    device = next(model.parameters()).device
    num_samples = min(max_samples, len(samples))
    for sample in samples[:num_samples]:
        input_ids = tokenizer(sample["input"], return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids)[0]
        prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
        reference = sample["output"].split()
        hypothesis = prediction.split()
        bleu_score = sentence_bleu([reference], hypothesis)
        # Ensure bleu_score is a float, not a list
        if isinstance(bleu_score, list):
            bleu_score = float(bleu_score[0]) if bleu_score else 0.0
        total_score += bleu_score
    avg_bleu = round(total_score / num_samples, 4)
    print(f"BLEU score: {avg_bleu}")
    return avg_bleu
