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

config = {'models': 
            {'hf_masked_model': ['bert-base-cased']},
         }

model = load_models(config)[0]

text = ['the boy is outside and the girl is with him.', 
        'the girl']
data = model.get_aligned_words_predictabilities(text)
for batch in data:
    for word in batch:
        print(word.word, word.surp, word.prob)
