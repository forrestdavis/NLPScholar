from src.models.LM import LM
from src.models.hf_causal_model import HFCausalModel

tokenizer_config = {'tokenizer': 
                    {'hf': ['gpt2']}}


model = HFCausalModel('gpt2', tokenizer_config, getHidden=True)
text = ['the boy is happy.', 'the boy are happy.']
output = model.get_by_token_predictability(text)
print(output[0])
print(output[1])
print(model.get_by_sentence_perplexity(text))

