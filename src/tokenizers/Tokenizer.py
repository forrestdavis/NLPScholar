# Basic Tokenizer parent class
# Heavily Inspired by HuggingFace's Tokenizer
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

import torch
import numpy as np
import string
from typing import Union, Dict, List, Tuple, Optional

class Tokenizer: 

    def __init__(self, version: str, doLower: bool = False,
                 addPrefixSpace: bool = False,
                 addPadToken: bool = True):
        """ Initialize a tokenizer. 
        """
        self.tokenizername = version
        self.doLower = doLower
        self.addPrefixSpace = addPrefixSpace
        self.addPadToken = addPadToken

    def __repr__(self):
        return self.tokenizername
    def __str__(self):
        return self.tokenizername

    def __call__(self):
        raise NotImplementedError

    def all_special_tokens(self):
        """ List of all special tokens """ 
        return None

    def TokenIDIsPunct(self, token_id:int) -> bool:
        """ Whether the id is English puncutation

        Args:
            token_id (`int`): id of token in vocab
        Returns:
            `bool`: True if id is for string in python string punctuation
        """
        return self.convert_ids_to_tokens(token_id) in string.punctuation

    def convert_tokens_to_ids(self, 
          tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id
        (or a sequence of ids), using the vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to
                                            token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], 
                              skip_special_tokens: bool = False
                                ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a
        sequence of tokens, using the vocabulary and added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        """
        Converts a string into a sequence of tokens, using the tokenizer.  Split
        in words for word-based vocabulary or sub-words for sub-word-based
        vocabularies (BPE/SentencePieces/WordPieces). Takes care of added
        tokens.

        Args:
            text (`str`):
                The sequence to be encoded.

        Returns:
            `List[str]`: The list of tokens.
        """
        raise NotImplementedError

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
            ) -> Union[List[int], torch.Tensor, np.ndarray]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer
        and vocabulary.

        Args:
            text (`str` or `List[str]`): The first sequence to be encoded. This
                 can be a string or a list of strings (tokenized string using the
                `tokenize` method) 
            add_special_tokens (`bool`): Whether to add special tokens to
                 tokenization. Default is True.
            padding (`bool`): Whether to pad to the maximum length
                 if True. Default is False.
            truncation (`bool`, *optional*): Whether to truncate to the maximum length
                 if True. Default is False.
            max_length (`int`, *optional*): Optionally specify the maximum length
            return_tensors (`str`, *optional*): Optionally specify the return 
                 type. `pt` for torch and `np` for numpy. Default return type 
                 is python list or ints. 

        """
        raise NotImplementedError

    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor],
        skip_special_tokens: bool = False,
            ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and
        vocabulary with option to remove special tokens.  

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. 
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
        Returns:
            `str`: The decoded sentence.
        """
        raise NotImplementedError


    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], np.ndarray,
                         torch.Tensor],
        skip_special_tokens: bool = False,
            ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray,
                torch.Tensor]`): List of tokenized input ids. 
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        raise NotImplementedError

    def align_words_ids(self, text: Union[str, List[str]]) -> List[dict]: 
        """ Returns the alignment between token ids and words and words.
       
        Args:
            text (`Union[str, List[str]]`): Input string or batch of strings

        Returns:
            `List[dict]`: List with a dictionary for each batch. Dictionary has
                        two elements: mapping_to_words, which is a list with the
                        word number each token belongs to, and words, which is a
                        list of each token's word string. 
        """
        raise NotImplementedError
