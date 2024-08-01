from src.models.load_models import load_models

config = {'models': 
            {'hf_masked_model': ['bert-base-cased']}, 
          'tokenizers':
            {'hf': ['bert-base-cased']}, 
          'ignorePunct': None, 
          'device': 'mps'}

model = load_models(config)[0]

print(model.get_output('the boy is happy'))
print()
print(model.get_by_token_predictability('the boy is happy'))
"""
text = ['the boy is happy.', 'the boy are happy.']
output = model.get_by_token_predictability(text)
print(output[0])
print(output[1])

print()

aligned = model.get_aligned_words_predictabilities(text)
print(aligned)
print()

print(model.get_by_sentence_perplexity(text))
"""
