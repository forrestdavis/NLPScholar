# Basic child using AutoModelForMaskedLM HuggingFace
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .LM import LM
from ..utils.load_tokenizers import load_tokenizers
import torch
from transformers import AutoModelForMaskedLM, AutoConfig
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

        self.isMaskedModel = True
        self.PLL_type = "within_word_l2r"
        if 'PLL_type' in kwargs:
            self.PLL_type = kwargs['PLL_type']

        # Load tokenizer
        if tokenizer_config is None:
            tokenizer_config = {'tokenizers': {'hf_tokenizer': [modelname]}}
        tokenizer_config = {**tokenizer_config, **kwargs}
        self.tokenizer = load_tokenizers(tokenizer_config)[0]

        modelkwargs = {'pretrained_model_name_or_path': modelname,
            'trust_remote_code': True, 
            'output_hidden_states': self.getHidden}

        if self.precision == '16bit':
            modelkwargs['torch_dtype'] = torch.float16
            modelkwargs['low_cpu_mem_usage'] = True
        elif self.precision == '8bit':
            modelkwargs['load_in_8bit'] = True

        elif self.precision == '4bit':
            modelkwargs['load_in_4bit'] = True

        # If we are loading a new model to train, we need to grab the config
        if not self.loadPretrained:
            auto_config = AutoConfig.from_pretrained(
                            modelname, 
                            vocab_size = len(self.tokenizer), 
                            n_positions = self.maxSequenceLength,
                            max_position_embeddings = self.maxSequenceLength,
                            bos_token_id = self.tokenizer.bos_token_id, 
                            eos_token_id = self.tokenizer.eos_token_id,
                        )
            self.model = \
                    AutoModelForMaskedLM.from_config(auto_config).to(self.device)

        else:
            self.model = \
                    AutoModelForMaskedLM.from_pretrained(
                            **modelkwargs).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def test(self, texts: Union[str, List[str]], 
                   stride: int = None, 
                   pplOnly: bool = False):

        assert self.PLL_type in {'original', 'within_word_l2r'}, f"PLL" \
                f" metric {PLL_type} not supported"

        # batchify 
        if isinstance(texts, str):
            texts = [texts]

        max_length = self.tokenizer.model_max_length

        # Set default stride to half max length
        if stride is None:
            stride = max_length//2

        inputs_dict = self.tokenizer(texts, 
                                     padding=True,
                                     return_tensors='pt').to(self.device)

        # This could be refactored to assist with adding other ppl measures
        nlls = []
        logits = None
        seq_length = inputs_dict.input_ids.size(1)
        prev_end_loc = 0
        for begin_loc in range(0, seq_length, stride):
            end_loc = min(begin_loc + max_length, seq_length)
            # may be different from stride on last loop
            trg_len = end_loc - prev_end_loc  

            input_ids = inputs_dict.input_ids[:, begin_loc:end_loc].to(
                                                        self.device)
            attn_mask = inputs_dict.attention_mask[:,
                                               begin_loc:end_loc].to(
                                                   self.device)

            # For each batch, create version of input where each token is
            # MASK'd.  This will generate conditional probabilities and follows,
            # in spirit, Salazar et al. (2020)
            # https://aclanthology.org/2020.acl-main.240.pdf Kanishka Misra's
            # minicons: https://github.com/kanishkamisra/minicons 
            # Loop through batch and replace each element of the input with MASK
            # token add model output to logits. 

            # If you don't do this, you get a bug!
            MASK = torch.tensor(self.tokenizer.mask_token_id).to(self.device)

            for idx in range(input_ids.size(1)):
                masked_input_ids = input_ids.clone()
                masked_input_ids[:,idx] = MASK

                # Assert that mask has been inserted 
                assert masked_input_ids[0,idx] == self.tokenizer.mask_token_id

                # Make the target_ids, which will have -100 in all places not
                # masked
                target_ids = input_ids.clone()
                # Before the masked index
                target_ids[:,:idx] = -100
                # After the masked index
                if idx < input_ids.size(1) - 1:
                    target_ids[:,idx+1:] = -100
                # Check that the target for the mask is still there
                assert target_ids[0,idx] != -100

                # Following Kauf & Ivanova (2023)
                # https://arxiv.org/abs/2305.10588
                if self.PLL_type == 'within_word_l2r':
                    # For each batch, we look at the words after the target
                    # word, if it is part of the same word as the target word
                    # (souvenir -> so ##uven ##ir) mask it as well.
                    for batch_idx in range(input_ids.size(0)):
                        word_ids = inputs_dict.word_ids(batch_idx)
                        mask_word = word_ids[idx]
                        if mask_word is None:
                            target_ids[batch_idx, idx] = -100
                            
                        for k in range(idx+1, masked_input_ids.size(1)):
                            if (word_ids[k] != mask_word or 
                                words_ids[k] is None):
                                break
                            masked_input[batch_idx,k] = MASK

                masked_dict = {'input_ids': masked_input_ids, 
                               'attention_mask': attn_mask, 
                               'labels': target_ids}

                output = self.model(**masked_dict)
                # Convert loss to base 2
                if output.loss != 0:
                    nlls.append(output.loss/torch.log(torch.tensor(2.0)))

                # Only keep all the logits when wanted (saves memory on 
                # document level perplexity calculations)
                if not pplOnly:
                    if logits is None:
                        logits = output.logits[:, target_ids[0,:] != -100, :]
                    else:
                        logits = torch.cat((logits, 
                                output.logits[:, target_ids[0,:] != -100, :]),
                                           1)

        print(nlls)
        ppl = 2**(torch.stack(nlls).mean())

        # Mark last position without padding
        # this works because transformers tokenizer flags 
        # padding with an attention mask value of 0
        attn_mask = inputs_dict.attention_mask.to(torch.int)
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

        return {'input_ids': inputs_dict.input_ids, 
                'last_non_masked_idx': last_non_masked_idx, 
                'logits': logits, 'perplexity': ppl}




    @torch.no_grad()
    def get_output(self, texts: Union[str, List[str]]):
        
        assert self.PLL_type in {'original', 'within_word_l2r'}, f"PLL metric {PLL_type} not supported"

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
            if self.PLL_type == 'within_word_l2r':
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

