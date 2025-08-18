from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./models/amr_parsing/")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/amr_parsing/")

input_sentence = ("Nhập một câu tiếng Việt để chuyển thành sơ đồ AMR:").strip()
input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)[0]
prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
print("\n📌 Sơ đồ AMR dự đoán:\n")
print(prediction)
