from src.models.LM import LM
from src.models.auto_causal_model import HFCausalModel

tokenizer_config = {'tokenizer': 
                    {'hf': ['gpt2']}}


model = HFCausalModel('gpt2', tokenizer_config, halfPrecision=True)
print(model.get_output(['the boy', 'the boy are']))
