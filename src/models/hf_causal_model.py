# Basic child using AutoModelForCausalLM HuggingFace
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .LM import LM
from ..utils.load_tokenizers import load_tokenizers
import torch
from transformers import AutoModelForCausalLM
from typing import Union, Dict, List, Tuple, Optional
import sys

class HFCausalModel(LM):

    def __init__(self, modelname: str, 
                 tokenizer_config: dict, 
                 offset=True,
                 **kwargs):

        super().__init__(modelname, tokenizer_config, 
                         offset=offset,
                         **kwargs)

        # Load tokenizer
        if tokenizer_config is None:
            tokenizer_config = {'tokenizers': {'hf_tokenizer': [modelname]}}
        tokenizer_config = {**tokenizer_config, **kwargs}
        self.tokenizer = load_tokenizers(tokenizer_config)[0]

        if self.precision == '16bit':
            self.model = AutoModelForCausalLM.from_pretrained(modelname, 
                                            torch_dtype=torch.float16, 
                                           low_cpu_mem_usage=True, 
                                           output_hidden_states=self.getHidden, 
                                           trust_remote_code=True).to(self.device)
        elif self.precision == '8bit':
            self.model = AutoModelForCausalLM.from_pretrained(modelname,
                                                    output_hidden_states=self.getHidden,
                                                        trust_remote_code=True,
                                                        load_in_8bit=True).to(self.device)

        elif self.precision == '4bit':
            self.model = AutoModelForCausalLM.from_pretrained(modelname,
                                                    output_hidden_states=self.getHidden,
                                                        trust_remote_code=True,
                                                        load_in_4bit=True).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(modelname, 
                                           output_hidden_states=self.getHidden, 
                                           trust_remote_code=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
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
                             f" {MAX_LENGTH}\n")
            sys.exit(1)

        # Mark last position without padding
        # this works because transformers tokenizer flags 
        # padding with an attention mask value of 0
        attn_mask = attn_mask.to(torch.int)
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

        return {'input_ids': inputs, 'last_non_masked_idx': last_non_masked_idx, 
                'logits': self.model(**inputs_dict).logits}

    @torch.no_grad()
    def get_hidden_layers(self, texts):

        # ensure we have loaded a model version that returns hidden 
        if not self.getHidden:
            sys.stderr.write("You must set getHidden to True to get hidden"\
                             " layers\n")

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


        # Downcast for mps warning
        attn_mask = attn_mask.to(torch.int)

        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

        return {'input_ids': inputs, 'last_non_masked_idx': last_non_masked_idx, 
                'hidden': self.model(**inputs_dict).hidden_states}
    


