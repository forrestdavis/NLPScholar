# Basic language model parent class
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

import torch
from ..tokenizers.load_tokenizers import load_tokenizers
from typing import Union, Dict, List, Tuple, Optional
import sys

class LM:

    def __init__(self, modelname: str, 
                 tokenizer_config: dict, 
                 offset: bool = True, 
                 getHidden: bool = False,
                 halfPrecision: bool = False,
                 device: str = 'best'):

        self.modelname = modelname
        self.halfPrecision = halfPrecision
        self.getHidden = getHidden
        self.offset = offset

        # Set device
        self.device = device
        if self.device == 'best':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_built():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        self.device = torch.device(self.device)
        sys.stderr.write(f"Running on {self.device}\n")

        # Load tokenizer
        self.tokenizer = load_tokenizers(tokenizer_config)[0]

        self.model = None

    @torch.no_grad()
    def get_output(self, text: Union[str, List[str]]):
        raise NotImplementedError

    @torch.no_grad()
    def get_targeted_output(self, text: Union[str, List[str]]):
        raise NotImplementedError

    @torch.no_grad()
    def get_hidden_layers(self, text: Union[str, List[str]]):
        raise NotImplementedError
