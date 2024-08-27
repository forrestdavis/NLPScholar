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
            {'hf_causal_model': ['gpt2-large']},
         }

'''
config = {'models': 
            {'hf_masked_model': ['bert-base-cased']},
         }
'''

model = load_models(config)[0]

text = ['the boys are', 'the boy are outside.']
#text = ['the boy are outside.']
#print(model.test(text)['perplexity'])
print('perplexity', model.get_output(text)['perplexity'])
print()
text = ['the boys are', 'the boy are', 'are outside.']
output = model.get_aligned_words_predictabilities(text)
total = 0
for batch in output:
    for word in batch:
        print(word.word, word.surp)
        total += word.surp
print(2**(total/6))
#from datasets import load_dataset
#test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#text = "\n\n".join(test['text'])

#print(model.get_output(text, stride=512, pplOnly=True)['perplexity'])
#print(model.get_by_token_predictability(text))
'''
print()
by_sent = model.get_by_sentence_perplexity(text)
ppls = []
print(by_sent)
import math
ppl = (2*math.log2(by_sent['perplexity'][0]) + 4*math.log2(by_sent['perplexity'][1]))/6
ppl = 2**ppl
print(ppl)

print()
output = model.get_by_token_predictability(text)
ppl = 0
count = 0
for out in output:
    for o in out:
        if o['token_id'] == 50256:
            continue
        if o['surprisal'] == 0:
            continue
        print(o)
        ppl += o['surprisal']
        count += 1


print(2**(ppl/count))
'''
'''
input_ids = model.tokenizer(text, return_tensors='pt', padding=True).to(model.device)
loss = model.model(**input_ids, labels=input_ids['input_ids']).loss
print('loss', loss)
loss = loss/torch.log(torch.tensor(2.0))
print('loss base 2', loss)
print('loss base 2', loss, 'perplexity base 2', 2**loss)
'''
