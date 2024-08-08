# Basic child using AutoModelForTokenClassification HuggingFace
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .Classifier import Classifier
from ..utils.load_tokenizers import load_tokenizers
import torch
from transformers import AutoModelForTokenClassification
from typing import Union, Dict, List, Tuple, Optional
import sys

class HFTokenClassificationModel(Classifier):

    def __init__(self, modelname: str, 
                 tokenizer_config: dict, 
                 **kwargs):

        super().__init__(modelname, tokenizer_config, 
                         **kwargs)

        # Load tokenizer
        if tokenizer_config is None:
            tokenizer_config = {'tokenizers': {'hf_tokenizer': [modelname]}}

        tokenizer_config = {**tokenizer_config, **kwargs}
        self.tokenizer = load_tokenizers(tokenizer_config)[0]

        if self.precision == '16bit':
            self.model = \
                AutoModelForTokenClassification.from_pretrained(modelname, 
                                        torch_dtype=torch.float16, 
                                       low_cpu_mem_usage=True, 
                                       trust_remote_code=True).to(self.device)
        elif self.precision == '8bit':
            self.model = \
                AutoModelForTokenClassification.from_pretrained(modelname,
                                        trust_remote_code=True,
                                        load_in_8bit=True).to(self.device)

        elif self.precision == '4bit':
            self.model = \
                AutoModelForTokenClassification.from_pretrained(modelname,
                                        trust_remote_code=True,
                                        load_in_4bit=True).to(self.device)
        else:
            self.model = \
                AutoModelForTokenClassification.from_pretrained(modelname,
                                           trust_remote_code=True).to(self.device)

        self.model.eval()

        if self.id2label is None:
            self.id2label = self.model.config.id2label

    @torch.no_grad()
    def get_token_output(self, texts: Union[str, List[str]]):
        
        # batchify 
        if isinstance(texts, str):
            texts = [texts]

        MAX_LENGTH = self.tokenizer.model_max_length

        inputs_dict = self.tokenizer(texts, 
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt').to(self.device)
        inputs = inputs_dict['input_ids']
        attn_mask = inputs_dict['attention_mask']

        # Mark last position without padding
        # this works because transformers tokenizer flags 
        # padding with an attention mask value of 0
        attn_mask = attn_mask.to(torch.int)
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

        return {'input_ids': inputs, 'last_non_masked_idx': last_non_masked_idx, 
                'logits': self.model(**inputs_dict).logits}

