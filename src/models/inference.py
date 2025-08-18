from transformers import T5ForConditionalGeneration, T5Tokenizer


def predict_amr_from_input(model_dir):
    """
    Load a trained T5 model and tokenizer, then predict AMR for user input.
    Args:
        model_dir (str): Path to the trained model directory.
    """
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return
    input_sentence = input(
        "Nhập một câu tiếng Việt để chuyển thành sơ đồ AMR: "
    ).strip()
    if not input_sentence:
        print("❗ Vui lòng nhập một câu hợp lệ.")
        return
    input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids
    import torch

    device = (
        next(model.parameters()).device
        if hasattr(model, "parameters")
        else torch.device("cpu")
    )
    input_ids = input_ids.to(device)
    model.to(device)
    output_ids = model.generate(input_ids)[0]
    prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("\n📌 Sơ đồ AMR dự đoán:\n")
    print(prediction)
