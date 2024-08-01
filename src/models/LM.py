# Basic language model parent class
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

import torch
from ..tokenizers.load_tokenizers import load_tokenizers
from typing import Union, Dict, List, Tuple, Optional
import sys
from collections import namedtuple

WordPred = namedtuple('WordPred', 'word surp prob isSplit isUnk '\
                              'modelName tokenizername')

class LM:

    def __init__(self, modelname: str, 
                 tokenizer_config: dict, 
                 offset=True,
                 **kwargs):

        self.modelname = modelname
        self.offset = offset

        # Default values
        self.getHidden = False
        self.precision = None
        self.showSpecialTokens = False
        self.device = 'best' 
        for k, v in kwargs.items():
            setattr(self, k, v)

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
    def get_output(self, text: Union[str, List[str]]) -> dict:
        """ Returns model output for text

        Args:
            text (`Union[str, List[str]]`): A (batch of) strings.

        Returns:
            `dict`: Dictionary with input_ids, last_non_masked_idx, and logits.
                    input_ids are the input ids from the tokenizer.
                    last_non_masked_idx is the index right before padding starts
                    of shape batch_size. Logits has shape (batch_size, number of
                    tokens, vocab_size).
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_hidden_layers(self, text: Union[str, List[str]]) -> dict:
        """ Returns model hidden layer representations for text

        Args:
            text (`Union[str, List[str]]`): A (batch of) strings.

        Returns:
            `dict`: Dictionary with input_ids, last_non_masked_idx, and
                    hidden_layers.
                    input_ids are the input ids from the tokenizer.
                    last_non_masked_idx is the index right before padding starts
                    of shape batch_size. hidden_layers is a tuple of length
                    number of layers (including embeddings). Each element has
                    shape (batch_size, number of tokens, hidden_size).
        """

        raise NotImplementedError

    @torch.no_grad()
    def convert_to_predictability(self, logits:torch.Tensor) -> torch.Tensor:
        """ Returns probabilities and surprisals from logits

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
        surprisals = -(log_probs/torch.log(torch.tensor(2.0)))
        return probabilities, surprisals

    @torch.no_grad()
    def get_by_token_predictability(self, text: Union[str, List[str]]
                                                     ) -> list: 
        """ Returns predictability measure of each token for inputted text.
           Note that this requires `get_output`.

        Args:
            text (`Union[str, List[str]]`): A (batch of) strings.

        Returns:
            `list`: List of lists for each batch comprised of dictionaries per
                    token. Each dictionary has token_id, probability, surprisal.
                    Note: the padded output is included.
        """
        output = self.get_output(text)
        input_ids = output['input_ids']
        last_non_masked_idx = output['last_non_masked_idx']
        
        by_token_probabilities, by_token_surprisals = \
                    self.convert_to_predictability(output['logits'])

        # If predictions are offset (as with causal LMs) then shift targets
        # using a zero vector for surprisal and a ones vector for probability.
        # This has the added benefit of making the first prediction value zero
        # for surprisal and one for probability.
        if self.offset:
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
            last_non_masked_idx = last_non_masked_idx + 1

        by_token_probabilities = by_token_probabilities.gather(-1,
                                                      input_ids.unsqueeze(2)).squeeze(-1)
        by_token_surprisals = by_token_surprisals.gather(-1,
                                                      input_ids.unsqueeze(2)).squeeze(-1)

        # For each batch, zip together input ids and predictability measures
        # into dictionaries 
        data = []
        for i in range(by_token_surprisals.size(0)):
            row = []
            for group in zip(input_ids[i, :], 
                        by_token_probabilities[i,:], 
                        by_token_surprisals[i,:],
                             strict=True):
                row.append({'token_id': int(group[0]), 
                            'probability': float(group[1]), 
                            'surprisal': float(group[2])})
            data.append(row)
        return data

    @torch.no_grad()
    def get_by_sentence_perplexity(self, text: Union[str, List[str]]
                                  ) -> List[Tuple[str, float]]:
        """ Returns perplexity of each sentence for inputted text.
           Note that this requires that you've implemented `get_output`.

           PPL for autoregressive models is defined as: 
            .. math::
                2^{-\\frac{1}{N} \\sum_{i}^{N} log p_{\\theta}(x_i|x_{<i})

            perplexity for bidirectional models uses psuedo-likelihoods
            (Salazar et al., 2020 https://aclanthology.org/2020.acl-main.240/)

        Args: 
            text (`Union[str, List[str]]`): A (batch of) strings.

        Returns:
            `List[Tuple[str, float]]`: List of the tuples of each string and
                perplexity. Padding, the first token for causal models,  and
                some special tokens are ignored in the calculation.  
        """

        # batchify
        if isinstance(text, str):
            text = [text]
        by_token_measures = self.get_by_token_predictability(text)
        return_data = []
        for idx in range(len(by_token_measures)):
            total_surprisal = 0
            length = 0
            # If offset, the first token should be ignored
            if self.offset:
                length = -1
            for token_measure in by_token_measures[idx]:
                if self.tokenizer.IsSkipTokenID(token_measure['token_id']):
                    continue
                total_surprisal += token_measure['surprisal']
                length += 1
            avg_surprisal = total_surprisal/length
            ppl = 2**avg_surprisal
            return_data.append((text[idx], ppl))
        return return_data

    @torch.no_grad()
    def get_aligned_words_predictabilities(self, text: Union[str, List[str]], 
                                          ) -> List[List[WordPred]]:
        """ Returns predictability measures of each word for inputted text.
           Note that this requires `get_output`.

        WordPred Object: 
            word: word in text
            surp: surprisal of word (total surprisal)
            prob: probability of word (joint probability)
            isSplit: whether it was subworded 
            isUnk: whether it is or contains an unk token
            modelName: name of language model 
            tokenizername: name of tokenizer used

        Args:
            text (`Union[str, List[str]]`): A (batch of) strings.

        Returns:
            `List[List[WordPred]]`: List of lists for each batch comprised of
                                    WordPred named tuple objects. 
        """

        all_data = []
        batched_token_predicts = self.get_by_token_predictability(text)
        batched_alignments = self.tokenizer.align_words_ids(text)
        for token_predicts, alignments_words in zip(batched_token_predicts,
                                                    batched_alignments,
                                                    strict=True):

            alignments = alignments_words['mapping_to_words']
            words = alignments_words['words']

            prob = 1
            surp = 0
            isUnk = False
            sentence_data = []
            for idx, (measure, alignment, word) in enumerate(zip(token_predicts,
                                                                 alignments,
                                                                 words,
                                                                 strict=True)):
                isSplit = False
                prob *= measure['probability']
                surp += measure['surprisal']
                if self.tokenizer.IsUnkTokenID(measure['token_id']):
                    isUnk = True
                # This is a special token (e.g., [CLS], [PAD])
                if word is None:
                    # Surface only when requested (still ignoring pad)
                    if (self.showSpecialTokens and 
                       self.tokenizer.pad_token_id != measure['token_id']):
                        word = self.tokenizer.convert_ids_to_tokens(
                            measure['token_id'])
                        sentence_data.append(WordPred(word, surp, prob, isSplit,
                                                      isUnk, self.modelname,
                                                      self.tokenizer.tokenizername))
                    prob = 1
                    surp = 0

                # Either the last word or the next word is new 
                elif ((idx == len(alignments) - 1) or 
                    (alignments[idx+1] != alignments[idx])):
                    if idx != 0 and alignments[idx-1] == alignments[idx]:
                        isSplit = True
                    sentence_data.append(WordPred(word, surp, prob, isSplit,
                                                  isUnk, self.modelname,
                                                  self.tokenizer.tokenizername))
                    prob = 1
                    surp = 0
                    isUnk = False
            all_data.append(sentence_data)
        return all_data

