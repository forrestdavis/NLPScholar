# Basic language model parent class
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

import torch
from typing import Union, Dict, List, Tuple, Optional

class LM:

    def __init__(self, modelname: str, 
                 tokenizer_config: dict, 
                 offset: bool = True, 
                 getHidden: bool = False,
                 halfPrecision: bool = False,
                 device: str = 'best'):

        self.modelname = modelname

        if self.device == 'best':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_built():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        self.device = torch.device(self.device)

    @torch.no_grad()
    def get_output(self, text: Union[str, List[str]]):
        raise NotImplementedError

    @torch.no_grad()
    def get_targeted_output(self, text: Union[str, List[str]]):
        raise NotImplementedError

    @torch.no_grad()
    def get_hidden_layers(self, text: Union[str, List[str]]):
        raise NotImplementedError

if __name__ == "__main__":
    print()
