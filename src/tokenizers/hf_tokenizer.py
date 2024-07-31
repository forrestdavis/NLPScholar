import torch
import transformers
import sys
from .Tokenizer import Tokenizer

class HFTokenizer(Tokenizer):
    """ Wraps Tokenizer class around HuggingFace AutoTokenizer """
    def __init__(self, version, doLower=False, addPrefixSpace=False, 
                addPadToken=True):
        super().__init__(version, doLower, addPrefixSpace, addPadToken)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(version,
                                            add_prefix_space=addPrefixSpace)
        self.convert_ids_to_tokens = self._tokenizer.convert_ids_to_tokens
        self.all_special_tokens = self._tokenizer.all_special_tokens
        self.model_max_length = self._tokenizer.model_max_length
        self.decode = self._tokenizer.decode
        self.batch_decode = self._tokenizer.batch_decode

        # deal with pad token 
        if self.addPadToken:
            if not self._tokenizer.pad_token:
                if self._tokenizer.eos_token_id is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    sys.stderr.write(f"Using {self._tokenizer.eos_token} as pad token\n")
                else:
                    sys.stderr.write(f"Need to specify a pad token\n")

    def IsSkipTokenID(self, token_id: int) -> bool:
        """ Whether the token id is for a skippable token. We consider cls, sep,
        and beginning of sentence tokens skippable. Note that eos is included. 

        Args: 
            token_id (`int`): Token id

        Returns:
            `bool`: Whether the token id is skippable
        """
        return (token_id == self._tokenizer.cls_token_id or 
                token_id == self._tokenizer.sep_token_id or 
                token_id == self._tokenizer.bos_token_id)

    def LowerCaseText(self, text):
        """ Lowercases text making sure to reintroduce special characters with
        the correct formatting
        NB. Probably slow, but i wanted to ensure lowercase happens
        """
        if self.doLower:
            if isinstance(text, str):
                text = text.lower()
                for special_token in self.all_special_tokens:
                    special_token_lower = special_token.lower()
                    text = text.replace(special_token_lower, special_token)
            elif isinstance(text, list):
                for idx, t in enumerate(text):
                    t = t.lower()
                    for special_token in self.all_special_tokens:
                        special_token_lower = special_token.lower()
                        t = t.replace(special_token_lower, special_token)
                        text[idx] = t
        return text 

    def __call__(
        self,
        text=None, 
        text_pair=None,
        text_target = None, 
        text_pair_target = None, 
        add_special_tokens = True,
        padding = False,
        truncation = None, 
        max_length = None,
        stride = 0,
        is_split_into_words = False,
        pad_to_multiple_of = None,
        return_tensors = None,
        return_token_type_ids = None,
        return_attention_mask = None,
        return_overflowing_tokens = False,
        return_special_tokens_mask = False,
        return_offsets_mapping = False,
        return_length = False,
        verbose = True,
        **kwargs):
        text = self.LowerCaseText(text)
        return self._tokenizer.__call__(text=text, 
        text_pair=text_pair,
        text_target = text_target, 
        text_pair_target = text_pair_target, 
        add_special_tokens = add_special_tokens,
        padding = padding,
        truncation = truncation, 
        max_length = max_length,
        stride = stride,
        is_split_into_words = is_split_into_words,
        pad_to_multiple_of = pad_to_multiple_of,
        return_tensors = return_tensors,
        return_token_type_ids = return_token_type_ids,
        return_attention_mask = return_attention_mask,
        return_overflowing_tokens = return_overflowing_tokens,
        return_special_tokens_mask = return_special_tokens_mask,
        return_offsets_mapping = return_offsets_mapping,
        return_length = return_length,
        verbose = verbose,
        **kwargs)
    def tokenize(self, text):
        text = self.LowerCaseText(text)
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        tokens = self.LowerCaseText(tokens)
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def encode(self, text, add_special_tokens = True,
        padding = False, truncation = False,
               max_length = None, return_tensors = None):
        text = self.LowerCaseText(text)
        return self._tokenizer.encode(text, 
                     add_special_tokens = add_special_tokens, 
                     padding=padding, truncation=truncation, 
                     max_length = max_length, return_tensors=return_tensors)
