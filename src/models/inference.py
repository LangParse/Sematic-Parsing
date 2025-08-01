from transformers import T5ForConditionalGeneration, T5Tokenizer


def predict_amr_from_input(model_dir):
    """
    Load a trained model and tokenizer, then predict AMR for user input.
    """
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    input_sentence = input(
        "Nhập một câu tiếng Việt để chuyển thành sơ đồ AMR: "
    ).strip()
    input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)[0]
    prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("\n📌 Sơ đồ AMR dự đoán:\n")
    print(prediction)
