from src.tokenizers.hf_tokenizer import HFTokenizer

tokenizer = HFTokenizer('bert-base-cased')
text = 'Rudolf is a red nosed reindeer'
print(tokenizer.align_words_ids(text))
#print(tokenizer.align_words_ids([text, text]))
