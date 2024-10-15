import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.load_models import load_models
import torch

config = {'models': 
            {'hf_masked_model': ['bert-base-cased']}, 
          'tokenizers':
            {'hf': ['bert-base-cased']}, 
          'device': 'mps'}

config = {'models': 
            {'hf_causal_model': ['gpt2']},
         }

'''
config = {'models': 
            {'hf_masked_model': ['bert-base-cased']},
         }
'''

model = load_models(config)[0]

from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = '\n\n'.join(test['text'])

text = [' '.join(text.split(' ')[:512]), ' '.join(text.split(' ')[512:1044])]

output = model.get_by_batch_perplexity(text)
for x in range(len(output['text'])):
    print(output['perplexity'][x], output['length'][x], output['text'][x])

'''
data = model.get_aligned_words_predictabilities(text)
for batch in data:
    for word in batch:
        print(word.word, word.surp, word.prob)
    print()
'''
