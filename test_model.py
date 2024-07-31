from src.models.LM import LM
from src.models.hf_causal_model import HFCausalModel
from src.models.hf_masked_model import HFMaskedModel

tokenizer_config = {'tokenizer': 
                    {'hf': ['bert-base-uncased']}}


model = HFMaskedModel('bert-base-uncased', tokenizer_config, getHidden=True,
                      precision='16bit')
text = ['the boy is happy.', 'the boy are happy.']
output = model.get_by_token_predictability(text)
print(output[0])
print(output[1])

print()

aligned = model.get_aligned_words_predictabilities(text, True)
print(aligned)
print()
