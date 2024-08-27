# Basic child using AutoModelForCausalLM HuggingFace
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .LM import LM
from ..utils.load_tokenizers import load_tokenizers
import torch
from transformers import AutoModelForCausalLM, AutoConfig
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

        self.isMaskedModel = False

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
                    AutoModelForCausalLM.from_config(auto_config).to(self.device)
        else:
            self.model = \
                    AutoModelForCausalLM.from_pretrained(**modelkwargs).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def get_output(self, texts: Union[str, List[str]], 
                   stride: int = None, pplOnly: bool = False):
        """ Sliding window is adapted from HuggingFace's example for fixed
        length models: https://huggingface.co/docs/transformers/en/perplexity 
        """

        # batchify 
        if isinstance(texts, str):
            texts = [texts]

        max_length = self.tokenizer.model_max_length
        max_length = 3

        # Set default stride to half max length
        if stride is None:
            stride = max_length//2
            stride = 2

        inputs_dict = self.tokenizer(texts, 
                                     padding=True,
                                     return_tensors='pt').to(self.device)

        all_input_ids = inputs_dict.input_ids
        all_attn_mask = inputs_dict.attention_mask
        # Mark last position without padding
        # this works because transformers tokenizer flags 
        # padding with an attention mask value of 0
        #attn_mask = inputs_dict.attention_mask.to(torch.int)
        last_non_masked_idx = torch.sum(all_attn_mask, dim=1) - 1
        # Create target ids and take special care to set pad token targets to
        # -100 so they are ignored by the loss. This is accomplished by applying
        # the attention mask and adding a created mask that either adds nothing
        # (for real targets) or sets it to -100
        ignore_mask = (all_attn_mask - 1)*100
        all_target_ids = all_input_ids.clone() * all_attn_mask + ignore_mask

        nlls = []
        logits = None
        seq_length = all_input_ids.size(1)
        prev_end_loc = 0
        total_targets_seen = 0
        for begin_loc in range(0, seq_length, stride):
            print('In the loop')
            end_loc = min(begin_loc + max_length, seq_length)
            # may be different from stride on last loop
            trg_len = end_loc - prev_end_loc  
            input_ids = all_input_ids[:, begin_loc:end_loc].to(self.device)
            attn_mask = all_attn_mask[:, begin_loc:end_loc].to(self.device)
            target_ids = all_target_ids[:, begin_loc:end_loc].to(self.device)
            # set repeated tokens to -100 
            target_ids[:, :-trg_len] = -100
            print(target_ids)
            print('input tokens',
                  self.tokenizer.convert_ids_to_tokens(input_ids[0]))
            print('input tokens',
                  self.tokenizer.convert_ids_to_tokens(input_ids[1]))

            new_inputs_dict = {'input_ids': input_ids, 
                               'attention_mask': attn_mask, 
                               'labels': target_ids}
            output = self.model(**new_inputs_dict)
            # Convert loss to base 2
            nlls.append(output.loss/torch.log(torch.tensor(2.0)))

            # Only keep all the logits when wanted (saves memory on document
            # level perplexity calculations)
            print('trg_len', trg_len)
            total_targets_seen += trg_len - 1
            print('logit shape', output.logits.shape)
            if not pplOnly:
                if logits is None:
                    logits = output.logits[:, -trg_len:, :]
                    print('added logits shape', output.logits[:, -trg_len:,
                                                              :].shape)
                else:
                    logits = torch.cat((logits, 
                                    output.logits[:, -trg_len:, :]),
                                       1)
                    print('added logits shape', output.logits[:, -trg_len:,
                                                              :].shape)
            # Keep track of the end locations for size of target region
            prev_end_loc = end_loc
            if end_loc == seq_length:
                break
            print()

        print('nlls')
        for nll in nlls:
            print('\t', float(nll))
        ppl = 2**(torch.stack(nlls).mean())
        #print(logits)
        print(logits.shape)


        return {'input_ids': inputs_dict.input_ids, 
                'last_non_masked_idx': last_non_masked_idx, 
                'logits': logits, 'perplexity': ppl}

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
    


