# Basic child using AutoModelForMaskedLM HuggingFace
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .LM import LM
from ..tokenizers.load_tokenizers import load_tokenizers
import torch
from transformers import AutoModelForMaskedLM
from typing import Union, Dict, List, Tuple, Optional
import sys

class HFMaskedModel(LM):

    def __init__(self, modelname: str, 
                 tokenizer_config: dict, 
                 offset=False,
                 **kwargs):

        super().__init__(modelname, tokenizer_config, 
                         offset=offset,
                         **kwargs)

        # Load tokenizer
        if tokenizer_config is None:
            tokenizer_config = {'tokenizers': {'hf': [modelname]}}
        self.tokenizer = load_tokenizers(tokenizer_config)[0]


        if self.precision == '16bit':
            self.model = AutoModelForMaskedLM.from_pretrained(modelname, 
                                            torch_dtype=torch.float16, 
                                           low_cpu_mem_usage=True, 
                                           output_hidden_states=self.getHidden, 
                                           trust_remote_code=True).to(self.device)
        elif self.precision == '8bit':
            self.model = AutoModelForMaskedLM.from_pretrained(modelname,
                                                    output_hidden_states=self.getHidden,
                                                        trust_remote_code=True,
                                                        load_in_8bit=True).to(self.device)

        elif self.precision == '4bit':
            self.model = AutoModelForMaskedLM.from_pretrained(modelname,
                                                    output_hidden_states=self.getHidden,
                                                        trust_remote_code=True,
                                                        load_in_4bit=True).to(self.device)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(modelname, 
                                           output_hidden_states=self.getHidden, 
                                           trust_remote_code=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_output(self, texts: Union[str, List[str]], 
                   PLL_type: str = "within_word_l2r"):
        
        assert PLL_type in {'original', 'within_word_l2r'}, f"PLL metric {PLL_type} not supported"

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

        logits = None
        # For each batch, create version of input where each token is MASK'd.
        # This will generate conditional probabilities and follows, in spirit,
        # Salazar et al. (2020) https://aclanthology.org/2020.acl-main.240.pdf
        # Kanishka Misra's minicons: https://github.com/kanishkamisra/minicons
        # Loop through batch and replace each element of the input with MASK
        # token add model output to logits. 

        # If you don't do this, you get a bug!
        MASK = torch.tensor(self.tokenizer.mask_token_id).to(self.device)

        for idx in range(inputs.shape[1]):
            masked_input = inputs.clone()
            masked_input[:,idx] = MASK

            assert masked_input[0,idx] == self.tokenizer.mask_token_id

            # Following Kauf & Ivanova (2023) https://arxiv.org/abs/2305.10588
            if PLL_type == 'within_word_l2r':
                # For each batch, we look at the words after the
                # target word, if it is part of the same word as
                # the target word (souvenir -> so ##uven ##ir)
                # mask it as well
                for j in range(inputs.shape[0]):
                    word_ids = inputs_dict.word_ids(j)
                    mask_word = word_ids[idx]
                    for k in range(idx+1, inputs.shape[1]):
                        if word_ids[k] != mask_word:
                            break
                        masked_input[j,k] = MASK

            masked_dict = {'input_ids': masked_input,
                           'attention_mask': attn_mask}

            out = self.model(**masked_dict).logits
            out = out[:,idx:idx+1,:]
            if logits is None:
                logits = out
            else:
                logits = torch.cat((logits, out), 1)

        return {'input_ids': inputs, 'last_non_masked_idx': last_non_masked_idx, 
                'logits': logits}

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

