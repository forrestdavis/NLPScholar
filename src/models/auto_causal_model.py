# Basic child using AutoModelForCausalLM HuggingFace
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .LM import LM
import torch
from transformers import AutoModelForCausalLM
from typing import Union, Dict, List, Tuple, Optional
import sys

class HFCausalModel(LM):

    def __init__(self, modelname: str, 
                 tokenizer_config: dict, 
                 offset: bool = True, 
                 getHidden: bool = False,
                 halfPrecision: bool = False,
                 device: str = 'best'):
        super().__init__(modelname, tokenizer_config, 
                         offset, getHidden, halfPrecision, 
                         device)

        if self.halfPrecision:
            self.model = AutoModelForCausalLM.from_pretrained(modelname, 
                                            torch_dtype=torch.float16, 
                                           low_cpu_mem_usage=True, 
                                           output_hidden_states=self.getHidden, 
                                           trust_remote_code=True).half().to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(modelname, 
                                           output_hidden_states=self.getHidden, 
                                           trust_remote_code=True).to(self.device)
        self.model.eval()

    def get_output(self, texts: Union[str, List[str]]):
        
        # batchify 
        if isinstance(texts, str):
            texts = [texts]

        MAX_LENGTH = self.tokenizer.model_max_length

        inputs_dict = self.tokenizer(texts, 
                                     padding=True,
                                     return_tensors='pt').to(self.device)
        inputs = inputs_dict['input_ids']
        attn_mask = inputs_dict['attention_mask']

        # if input is longer than maximum allowed stop 
        if inputs.shape[1] > MAX_LENGTH:
            sys.stderr.write(f"Input is {inputs.shape[1]} while max is"\
                             f"{MAX_LENGTH}\n")
            sys.exit(1)

        attn_mask = attn_mask.to(torch.int)
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
        return {'inputs': inputs, 'last_non_masked_idx': last_non_masked_idx, 
                'logits': self.model(**inputs_dict).logits}


