import torch
import transformers
from .Tokenizer import Tokenizer

class HFTokenizer(Tokenizer):
    """ Wraps Tokenizer class around HuggingFace AutoTokenizer """
    def __init__(self, version, doLower=False, addPrefixSpace=False):
        super().__init__(version, doLower, addPrefixSpace)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(version,
                                            add_prefix_space=addPrefixSpace)
        self.convert_ids_to_tokens = self._tokenizer.convert_ids_to_tokens
        self.all_special_tokens = self._tokenizer.all_special_tokens
        self.decode = self._tokenizer.decode
        self.batch_decode = self._tokenizer.batch_decode

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