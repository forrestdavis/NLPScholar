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

model = load_models(config)[0]

text = ['the boy is outside.']
print(model.get_by_sentence_perplexity(text))

input_ids = model.tokenizer(text, return_tensors='pt', padding=True).to(model.device)
loss = model.model(**input_ids, labels=input_ids['input_ids']).loss
print(loss)
loss = loss/torch.log(torch.tensor(2.0))
print(loss, 2**loss)

