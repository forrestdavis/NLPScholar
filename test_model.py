from src.models.load_models import load_models

config = {'models': 
            {'hf_masked_model': ['bert-base-cased']}, 
          'tokenizers':
            {'hf': ['bert-base-cased']}, 
          'device': 'mps'}

config = {'models': 
            {'hf_masked_model': ['bert-base-cased'],
            'hf_causal_model': ['gpt2-medium']},
         }

models = load_models(config)

for model in models:
    print(model, model.tokenizer)

import sys
sys.exit()

output = model.get_by_token_predictability('Rudolfson is happy.')[0]
for word in output:
    print(model.tokenizer.convert_ids_to_tokens(word['token_id']))
    print(word['surprisal'])


output = model.get_aligned_words_predictabilities(['Rudolfson is happy.'])
"""
                                               'the other day i went to the '\
                                                'store and saw'])
"""
for sentence in output:
    print('-'*80)
    for word in sentence:
        print(word.word)
        print('\t', word.isSplit)
        print('\t', word.prob)
        print('\t', word.surp)
    print('-'*80)
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
