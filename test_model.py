from src.models.load_models import load_models

config = {'models': 
            {'hf_causal_model': ['gpt2']}, 
          'tokenizers':
            {'hf': ['gpt2']}, 
          'ignorePunct': None, 
          'device': 'mps'}

model = load_models(config)[0]


text = ['the boy is happy.', 'the boy are happy.']
output = model.get_by_token_predictability(text)
print(output[0])
print(output[1])

print()

aligned = model.get_aligned_words_predictabilities(text)
print(aligned)
print()
