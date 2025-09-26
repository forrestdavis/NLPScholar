# Basic class for minimal pair evaluations (or TSE)
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .Evaluation import Evaluation
import pandas as pd

class MinimalPair(Evaluation):

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)
        self.NEEDS = ['sentid', 'sentence']

    def gather_token_output(self, LM):
        """ Returns the outputs and word alignments for a langaguage model for
            the experiment. 

        Args:
            LM (`LM`): An instance of a LM

        Returns:
            `list`: List of lists for each batch comprised of dictionaries per
                    token. Each dictionary has token_id, probability, surprisal.
                    Note: the padded output is included.
            `List[dict]`: List with a dictionary for each batch. Dictionary has
                        two elements: mapping_to_words, which is a list with the
                        word number each token belongs to, and words, which is a
                        list of each token's word string. 

        """
        sentences = self.data['sentence'].tolist()
        outputs = []
        word_alignments = []
        for batch_idx in range(0, len(sentences), self.batchSize):
            batch = sentences[batch_idx:batch_idx+self.batchSize]
            outputs.extend(LM.get_by_token_predictability(batch))
            word_alignments.extend(LM.tokenizer.align_words_ids(batch))
        return outputs, word_alignments

    def add_entries(self, entriesDict, outputs, word_alignments, LM):
        """ Adds entries to the dictionary of results.
        """
        sentences = self.data['sentence'].tolist()
        sentids = self.data['sentid'].tolist()

        for batch in range(len(sentences)):
            output = outputs[batch]
            sentid = sentids[batch]
            alignments = word_alignments[batch]['mapping_to_words']
            words = word_alignments[batch]['words']

            for measure, alignment, word in zip(output, alignments, words,
                                                strict=True):
                
                # Skip pad tokens 
                if LM.tokenizer.pad_token_id == measure['token_id']:
                    continue

                # Get token
                token = LM.tokenizer.convert_ids_to_tokens(measure['token_id'])

                if word is None:
                    continue

                # Figure out if punctuation 
                isPunct = False
                if LM.tokenizer.TokenIDIsPunct(measure['token_id']):
                    isPunct = True

                # Add to prediction data
                entriesDict['token'].append(token)
                entriesDict['sentid'].append(sentid)
                entriesDict['word'].append(word)
                entriesDict['wordpos'].append(alignment)
                entriesDict['model'].append(LM.modelname)
                entriesDict['tokenizer'].append(LM.tokenizer.tokenizername)
                entriesDict['punctuation'].append(isPunct)
                entriesDict['prob'].append(measure['probability'])
                entriesDict['surp'].append(measure['surprisal'])
        return 


    def evaluate(self):
        """ Evaluate MinimalPair and save result to predfpath. 

        The output has the following information. 
        - `token`: the subword token
        - `sentid`: the ID of the specific pair of sentences being compared
        - `word`: the word the subword token belongs to
        - `wordpos`: the specific word in the context sentence that the subword token belongs to
        - `model`: the name of the language model
        - `tokenizer`: the name of the tokenizer
        - `punctuation`: is the token punctuation?
        - `prob`: the probability of the token given the context
        - `surp`: the surprisal of the token given the context; log base 2

        """
        # Load data 
        assert self.datafpath is not None, "Missing a datafpath value\n"
        assert self.predfpath is not None, "Missing a predfpath value\n"

        self.data = self.load_cond()
        if self.checkFileColumns:
            self.columnCheck()

        predictData = {'token': [], 
                       'sentid': [], 
                       'word': [],
                       'wordpos': [], 
                       'model': [],
                       'tokenizer': [],
                       'punctuation': [],
                       'prob': [], 
                       'surp': [],
                      }
        for LM in self.Models:
            outputs, word_alignments = self.gather_token_output(LM)
            self.add_entries(predictData, outputs, word_alignments, LM)

        predictData = pd.DataFrame.from_dict(predictData)
        self.save_evaluation(predictData)

    def interact(self):
        """ Run interactive mode with MinimalPair. User is asked to input a text
        and the by-word predictability measures are given.

        Prints:
            `word`: The word given by tokenizer 
            `Split`: Whether the word was split 
            `Unk`: Whether the word is an unk token
            `ModelName`: Name of the model (shortened to fit on screen)
            `surp`: Surprisal of the word (joint over subword tokens)
            `prob`: Probability of the word (joint over subword tokens)
        """

        word = "word" + ' '*16
        split = "Split" 
        unk = "Unk"
        model = "ModelName" + ' '*11
        surp = "surp" + ' '*4
        prob = "prob" + ' '*6
        header = f"{word} | {split} | {unk} | {model} | {surp} | {prob}"

        # Only consider the first model
        if self.loadAll:
            LM = self.Models[0]
        else:
            LM = next(self.Models)

        while True:
            sent = input('text (or STOP): ').strip()
            if sent == 'STOP':
                break
            print(header)
            print('-'*len(header))
            output = LM.get_aligned_words_predictabilities(sent)[0]
            for word in output:
                surp = round(word.surp, 3)
                prob = round(word.prob, 5)
                print_out = f"{word.word: <20} | {word.isSplit:5} | {word.isUnk:3} | {word.modelName.split('/')[-1]: <20} | {surp: >8} | {prob: >10}"
                print(print_out)        
