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
                 addPrefixSpace: bool = False,):
        """ Initialize a tokenizer. 
        """
        self.version = version
        self.doLower = doLower
        self.addPrefixSpace = addPrefixSpace

    def __call__(self):
        raise NotImplementedError

    def align_words_ids(self, text: Union[str, List[str]], 
                        lang: str = 'en', 
                       ) -> Union[Tuple[List[str], List[int]], 
                                  List[Tuple[List[str], List[int]]]]:
        """ Aligns words in text input with their ids in the input 
       
        Args:
            text (`Union[str, List[str]]`): Input string or batch of strings
            lang (`str`): Language to split with. 'en' is whitespace and is
                    efault. 

        Returns:
            `Union[Tuple[List[str], List[int]], List[Tuple[List[str],
            List[int]]]]`: Tuple of the words and the ids per word or a list of
                tuples for the batch. 
        """
        input_ids = self.__call__(text, return_offsets_mapping=True)

        if isinstance(text, str):
            return self._align_words_ids(text, input_ids['offset_mapping'], 
                                        input_ids['input_ids'], lang)
        elif isinstance(text, list):
            data = []
            for idx, t in enumerate(text):
                in_ids = input_ids['input_ids'][idx]
                offset_mapping = input_ids['offset_mapping'][idx]
                data.append(self._align_words_ids(t, offset_mapping, 
                                             in_ids, lang))
            return data
        return None

    def _align_words_ids(self, text: str, offset_mapping: List[Tuple[int]], 
                         input_ids: List[int], 
                         lang: str) -> Tuple[List[str], List[int]]:
        """ Aligns words in a sentence with their ids in the input 
       
        Args: text (`str`): String representing text
            offset_mapping (`List[Tuple[int]]`): List of tuples representing the
                    start and end position of the token in the string
            input_ids (`List[int]`): List of input ids from tokenization
            lang (`str`): Language to split with. 'en' is whitespace

        Returns:
            `Tuple[List[str], List[int]]`: Tuple of the words and the ids per
                    word.
        """
        if lang == 'en':
            chunks = text.split()
        else:
            import sys
            sys.stderr.write(f"{lang} has no splitting rule\n")
            sys.exit(1)

        chunk = chunks.pop(0)

        ids = []
        words = []

        string = ''
        sub_ids = []
        for idx, region in enumerate(offset_mapping):
            # Is an added special token (e.g., [CLS])
            if region[0] == 0 and region[1] == 0:
                continue
            region_str = text[region[0]:region[1]].strip()
            sub_ids.append(input_ids[idx])
            string += region_str 
            if string == chunk:
                words.append(string)
                ids.append(sub_ids)
                sub_ids = []
                string = ''
                if chunks:
                    chunk = chunks.pop(0)

        assert len(words) == len(ids) == len(text.split())
        return (words, ids)

    def all_special_tokens(self):
        """ List of all special tokens """ 
        return None

    def id_is_punct(self, token_id:int) -> bool:
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

if __name__ == "__main__":
    tokenizer = Tokenizer('ye')
