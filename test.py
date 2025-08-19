# from src.amr_tokenizer import AMRTokenizer
#
# amr_tok = AMRTokenizer(
#     model_name="google/long-t5-tglobal-base",
#     max_length_input=256,
#     max_length_output=512,
# )
#
# text = "gặp những đứa con của quê hương này , tôi bị bất ngờ bởi sức sống mãnh liệt ."
#
# inputs = amr_tok.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
#
# print(inputs)
# # Decode the input_ids to get the segmented text
# print(amr_tok.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
#
# # If you want to segment the text, you can use the tokenizer's `tokenize` method
# tokens = amr_tok.tokenizer.tokenize(text)
# print("Segmented tokens:", tokens)
