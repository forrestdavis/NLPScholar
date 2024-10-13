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
                            n_positions = self.maxTrainSequenceLength,
                            max_position_embeddings = self.maxTrainSequenceLength,
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

        # Set stride to half max length if you haven't specified it
        if self.stride is None:
            self.stride = int(self.tokenizer.model_max_length//2)

    @torch.no_grad()
    def get_by_token_predictability(self, texts: Union[str, List[str]]):
        
        assert self.PLL_type in {'original', 'within_word_l2r'}, f"PLL metric {PLL_type} not supported"

        # batchify 
        if isinstance(texts, str):
            texts = [texts]

        MAX_LENGTH = self.tokenizer.model_max_length - 2

        # Note: Special tokens are not added, instead the individual 
        # logit calls will add special tokens
        inputs_dict = self.tokenizer(texts, 
                                     padding=True,
                                     return_tensors='pt', 
                                    add_special_tokens=False).to(self.device)
        input_ids = inputs_dict['input_ids']
        attn_mask = inputs_dict['attention_mask']

        seq_len = input_ids.size(1)

        if seq_len > MAX_LENGTH:
            sys.stderr.write(f"Sequence length: {seq_len} is larger than "\
                             f"maximum sequence length allowed by the "\
                             f"model: {MAX_LENGTH} (without seperating tokens"\
                             f"). Using a stride of "\
                             f"{self.stride} in calculating token "\
                             f"predictabilities.\n")

        # Adapted from HuggingFace's perpelxity for fixed length models 
        # https://huggingface.co/docs/transformers/main/en/perplexity
        prev_end_loc = 0
        # Set up data
        data = []
        for _ in range(input_ids.size(0)):
            data.append([])
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + MAX_LENGTH, seq_len)
            trg_len = end_loc - prev_end_loc
            strided_input_ids = input_ids[:,begin_loc:end_loc].to(self.device)

            # reconvert to text 
            strided_text = []
            for idx in range(strided_input_ids.size(0)):
                strided_text.append(self.tokenizer.decode(
                                    strided_input_ids[idx, :], 
                                    skip_special_tokens=True))

            logits = self.get_logits(strided_text)['logits']

            by_token_probabilities, by_token_surprisals = \
                    self.convert_to_predictability(logits)

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
            `dict`: Dictionary with input_ids and logits.
                    input_ids are the input ids from the tokenizer.
                    Logits has shape (batch_size, number of
                    tokens, vocab_size).
        """
        
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

            # Skip CLS/SEP
            if (inputs[0, idx] in [self.tokenizer.cls_token_id, 
                                   self.tokenizer.sep_token_id]):
                continue

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

        return {'input_ids': inputs, 'logits': logits}

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

        return {'input_ids': inputs, 
                'hidden': self.model(**inputs_dict).hidden_states}

