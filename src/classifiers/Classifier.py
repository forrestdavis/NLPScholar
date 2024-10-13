# Basic classification parent class
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

import torch
from ..utils.load_tokenizers import load_tokenizers
from typing import Union, Dict, List, Tuple, Optional
import sys
from collections import namedtuple

WordPred = namedtuple('WordPred', 'word surp prob isSplit isUnk '\
                              'modelName tokenizername')

class Classifier:

    def __init__(self, modelname: str, 
                 tokenizer_config: dict, 
                 **kwargs):

        self.modelname = modelname

        # Default values
        self.precision = None
        self.device = 'best' 
        self.id2label = None
        self.label2id = None
        self.loadPretrained = True
        self.numLabels = None
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.device == 'best':
            if torch.backends.mps.is_built():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        self.device = torch.device(self.device)
        sys.stderr.write(f"Running on {self.device}\n")

        self.tokenizer = None
        self.model = None

    def __repr__(self):
        return self.modelname
    def __str__(self):
        return self.modelname

    @torch.no_grad()
    def get_text_output(self, text: Union[str, List[str]], 
                        pair: Union[str, List[str]] = None) -> dict:
        """ Returns model output for text

        Args:
            text (`Union[str, List[str]]`): A (batch of) strings.
            pair (`Union[str, List[str]]`): A *optional* (batch of) strings to
                                            use as a second sentence. 

        Returns:
            `dict`: Dictionary with input_ids, last_non_masked_idx, and logits.
                    input_ids are the input ids from the tokenizer.
                    last_non_masked_idx is the index right before padding starts
                    of shape batch_size. Logits has shape (batch_size, number of
                    labels).
        """
        raise NotImplementedError

    @torch.no_grad()
    def convert_to_probability(self, logits:torch.Tensor) -> torch.Tensor:
        """ Returns probabilities from logits

        Args:
            logits (`torch.Tensor`): logits with shape (batch_size,
                                            number of tokens, vocab_size),
                                        as in output of get_output()

        Returns:
            `torch.Tensor`: probabilities and surprisals (base 2) each 
                    with shape (batch_size, number of tokens, vocab_size)
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probs)
        return probabilities

    def get_text_predictions(self, text: Union[str, List[str]], 
                             pair: Union[str, List[str]] = None) -> List[dict]:
        """ Returns model predicted labels for text

        Args:
            text (`Union[str, List[str]]`): A (batch of) strings.
            pair (`Union[str, List[str]]`): A *optional* (batch of) strings to
                                            use as a second sentence. 

        Returns:
            `List[dict]`: A list with dictionaries per batch. Each dictionary
                          has the label, probability, and id.  
        """

        output = self.get_text_output(text, pair)
        probabilities, predictions = \
                self.convert_to_probability(output['logits']).max(-1)

        data = []
        for probability, prediction in zip(probabilities, predictions):

            prediction = prediction.item() 
            probability = probability.item()
            data.append({'label': self.id2label[prediction], 
                         'probability': probability, 
                         'id': prediction})
        return data

    def get_by_token_predictions(self, text: Union[str, List[str]]): 
        output = self.get_token_output(text)
        probabilities, predictions = \
                self.convert_to_probability(output['logits']).max(-1)
        
        data = []
        for batch in range(probabilities.size(0)):
            batch_data = []
            for token_id, probability, prediction in zip(
                                                output['input_ids'][batch],
                                                probabilities[batch],
                                               predictions[batch], 
                                              strict=True):
                token_id = token_id.item()
                prediction = prediction.item() 
                probability = probability.item()
                batch_data.append({
                            'token_id': token_id,
                            'label': self.id2label[prediction], 
                             'probability': probability, 
                             'id': prediction})
            data.append(batch_data)
        return data
