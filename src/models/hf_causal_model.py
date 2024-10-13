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
                            n_positions = self.maxTrainSequenceLength,
                            max_position_embeddings = self.maxTrainSequenceLength,
                            bos_token_id = self.tokenizer.bos_token_id, 
                            eos_token_id = self.tokenizer.eos_token_id,
                        )
            self.model = \
                    AutoModelForCausalLM.from_config(auto_config).to(self.device)
        else:
            self.model = \
                    AutoModelForCausalLM.from_pretrained(**modelkwargs).to(self.device)

        self.model.eval()

        # Set stride to half max length if you haven't specified it
        if self.stride is None:
            self.stride = int(self.tokenizer.model_max_length//2)

    @torch.no_grad()
    def get_by_token_predictability(self, texts: Union[str, List[str]]
                                                     ) -> list: 
        # batchify 
        if isinstance(texts, str):
            texts = [texts]

        MAX_LENGTH = self.tokenizer.model_max_length

        inputs_dict = self.tokenizer(texts, 
                                     padding=True,
                                     return_tensors='pt').to(self.device)
        input_ids = inputs_dict['input_ids']
        attn_mask = inputs_dict['attention_mask']

        seq_len = input_ids.size(1)

        # Adapted from HuggingFace's perpelxity for fixed length models 
        # https://huggingface.co/docs/transformers/main/en/perplexity
        prev_end_loc = 0
        data = []
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + MAX_LENGTH, seq_len)
            trg_len = end_loc - prev_end_loc
            strided_input_ids = input_ids[:,begin_loc:end_loc].to(self.device)
            strided_attn_mask = attn_mask[:,begin_loc:end_loc].to(self.device)

            strided_input = {'input_ids': strided_input_ids, 
                             'attention_mask': strided_attn_mask}

            logits = self.model(**strided_input).logits

            by_token_probabilities, by_token_surprisals = \
                    self.convert_to_predictability(logits)
            # Shift targets because predictions are offest using a zero vector
            # for surprisal and a ones vector for probability. This has the
            # added benefit of making the first prediction value zero for
            # surprisal and one for probability
            by_token_probabilities = torch.cat((torch.ones(
                                        by_token_probabilities.size(0), 1,
                                     by_token_probabilities.size(-1)).to(self.device),
                                         by_token_probabilities), 
                                         dim = 1)
            by_token_surprisals = torch.cat((torch.zeros(
                                                by_token_surprisals.size(0), 1, 
                                                by_token_surprisals.size(-1)).to(self.device),
                                         by_token_surprisals), 
                                         dim = 1)

            # Get by token measures 
            by_token_probabilities = by_token_probabilities.gather(-1,
                                               strided_input_ids.unsqueeze(2)).squeeze(-1)
            by_token_surprisals = by_token_surprisals.gather(-1,
                                                strided_input_ids.unsqueeze(2)).squeeze(-1)

            # Extract the actual targets (those predictions which are new)
            by_token_probabilities = by_token_probabilities[:, -trg_len:]
            by_token_surprisals = by_token_surprisals[:, -trg_len:]
            strided_input_ids = strided_input_ids[:, -trg_len:]

            # For each batch, zip together input ids and predictability measures
            # into dictionaries 
            for i in range(by_token_surprisals.size(0)):
                row = []
                for group in zip(strided_input_ids[i, :], 
                            by_token_probabilities[i,:], 
                            by_token_surprisals[i,:],
                                 strict=True):
                    row.append({'token_id': int(group[0]), 
                                'probability': float(group[1]), 
                                'surprisal': float(group[2])})
                if data == []:
                    data.append(row)
                else:
                    data[i].extend(row)

            prev_end_loc = end_loc
            # Wrap up if you've reached the end with the current chunk
            if end_loc == seq_len:
                break

        return data

    @torch.no_grad()
    def get_logits(self, texts: Union[str, List[str]]):
        """ Returns model logits for text

        Args:
            texts (`Union[str, List[str]]`): A (batch of) strings.

        Returns:
            `dict`: Dictionary with input_ids, last_non_masked_idx, and logits.
                    input_ids are the input ids from the tokenizer.
                    last_non_masked_idx is the index right before padding starts
                    of shape batch_size. Logits has shape (batch_size, number of
                    tokens, vocab_size).
        """
        
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
    


