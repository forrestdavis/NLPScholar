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

        modelkwargs = {'pretrained_model_name_or_path': modelname,
            'trust_remote_code': True}

        if self.precision == '16bit':
            modelkwargs['torch_dtype'] = torch.float16
            modelkwargs['low_cpu_mem_usage'] = True
        elif self.precision == '8bit':
            modelkwargs['load_in_8bit'] = True

        elif self.precision == '4bit':
            modelkwargs['load_in_4bit'] = True

        # If we are loading a new model to finetune we need to specify the
        # number of labels for our classification head.
        # Note that this assumes the base model has already been trained.
        if not self.loadPretrained:
            # Add the number of labels 
            assert self.numLabels is not None, "You must specify numLabels" \
                    " when loading a model for finetuning"
            modelkwargs['num_labels'] = self.numLabels

        # Load label mappings 
        if self.id2label is not None:
            modelkwargs['id2label'] = self.id2label
            if self.label2id is None:
                self.label2id = {}
                for i, l in self.id2label.items():
                    self.label2id[l] = i
            modelkwargs['label2id'] = self.label2id

        self.model = \
                AutoModelForTokenClassification.from_pretrained(
                    **modelkwargs).to(self.device)

        self.model.eval()

        if self.id2label is None:
            self.id2label = self.model.config.id2label
        if self.label2id is None:
            self.label2id = self.model.config.label2id

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

