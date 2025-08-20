
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu

def bleu_on_samples(model, tokenizer, samples: List[Dict[str, str]], limit: int = 100) -> float:
    model.eval()
    total = 0.0
    use = min(limit, len(samples)) if limit else len(samples)
    for idx, s in enumerate(samples[:use]):
        input_ids = tokenizer(s["input"], return_tensors="pt").input_ids.to(model.device)
        output_ids = model.generate(input_ids, max_length=512)[0]
        pred = tokenizer.decode(output_ids, skip_special_tokens=True)
        reference = s["output"].split()
        hypothesis = pred.split()
        total += sentence_bleu([reference], hypothesis)
    return round(total / use, 4) if use else 0.0
