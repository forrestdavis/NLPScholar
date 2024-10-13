import torch
import transformers
import sys
from .Tokenizer import Tokenizer

class HFTokenizer(Tokenizer):
    """ Wraps Tokenizer class around HuggingFace AutoTokenizer """
    def __init__(self, tokenizername: str, 
                 **kwargs):

        super().__init__(tokenizername, **kwargs)
        self._tokenizer = \
            transformers.AutoTokenizer.from_pretrained(tokenizername,
                                        add_prefix_space=self.addPrefixSpace)
        self.convert_ids_to_tokens = self._tokenizer.convert_ids_to_tokens
        self.all_special_tokens = self._tokenizer.all_special_tokens
        self.model_max_length = self._tokenizer.model_max_length
        self.decode = self._tokenizer.decode
        self.batch_decode = self._tokenizer.batch_decode

        # deal with pad token 
        if not self._tokenizer.pad_token:
            if self.addPadToken == True or self.addPadToken == 'eos_token':
                if self._tokenizer.eos_token_id is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    sys.stderr.write(f"Using {self._tokenizer.eos_token} as pad token\n")
                else:
                    self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    sys.stderr.write(f"There is no eos token for the "\
                                     "tokenizer. Creating a new token to use "\
                                     "as a pad token, [PAD].\n")
            elif self.addPadToken:
                token_id = self.encode(self.addPadToken)
                if len(token_id) != 1:
                    sys.stderr.write(f"{self.addPadToken} is not a single " \
                                     "token so it cannot be a pad token\n")
                    sys.exit(1)

                self._tokenizer.pad_token = self.addPadToken

    def __len__(self):
        return len(self._tokenizer)

    @property 
    def bos_token_id(self) -> int:
        return self._tokenizer.bos_token_id

    @property 
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id
                    
    @property 
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property 
    def mask_token_id(self) -> int:
        return self._tokenizer.mask_token_id

    @property 
    def sep_token_id(self) -> int:
        return self._tokenizer.sep_token_id

    @property 
    def cls_token_id(self) -> int:
        return self._tokenizer.cls_token_id

    def IsSkipTokenID(self, token_id: int) -> bool:
        """ Whether the token id is for a skippable token. We consider cls, sep,
        and beginning of sentence tokens skippable. Note that eos is not included. 

        Args: 
            token_id (`int`): Token id

        Returns:
            `bool`: Whether the token id is skippable
        """
        return (token_id == self._tokenizer.cls_token_id or 
                token_id == self._tokenizer.sep_token_id or 
                token_id == self._tokenizer.bos_token_id)

    def IsUnkTokenID(self, token_id: int) -> bool:
        """ Whether the token id is for an unk token. 

        Args: 
            token_id (`int`): Token id

        Returns:
            `bool`: Whether the token id is unk
        """
        return token_id == self._tokenizer.unk_token_id

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

    def align_words_ids(self, text, add_special_tokens=False):
        # batchify 
        if isinstance(text, str):
            text = [text]
        text = self.LowerCaseText(text)
        encoded = self._tokenizer(text, padding=True, 
                                  add_special_tokens=False)
        data = []
        for batch in range(len(encoded['input_ids'])):
            mapping = encoded.word_ids(batch)
            words = []
            for element in mapping: 
                word = None
                if element is not None: 
                    start, end = encoded.word_to_chars(batch, element)
                    word = text[batch][start:end]
                words.append(word)
            data.append({'mapping_to_words': mapping, 
                         'words': words})
        return data


