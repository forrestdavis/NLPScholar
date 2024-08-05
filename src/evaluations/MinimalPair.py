# Basic class for minimal pair evaluations (or TSE)
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..utils.load_models import load_models, yield_models
import sys
import pandas as pd

class MinimalPair:

    def __init__(self, config: dict, 
                **kwargs):

        # Default values
        self.loadAll = False
        self.checkFileFormat = True
        self.batchSize = 1
        self.verbose = True
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Check for necessary things 
        assert 'predfpath' in config, "Must pass in predfpath"
        assert 'condfpath' in config, "Must pass in condfpath"
        self.predfpath = config['predfpath']
        self.condfpath = config['condfpath']

        # Load models
        if self.loadAll:
            if self.verbose:
                sys.stderr.write('Loading all models to memory...\n')
            self.LMs = load_models(config)
        else:
            self.LMs = yield_models(config)

        # Load data 
        self.data = self.load_cond()
        if self.checkFileFormat:
            self.formatCheck()


    def load_cond(self):
        """ Load condition file 

        Returns:
            `pd.DataFrame`: pandas dataframe of data
        """
        if self.verbose:
            sys.stderr.write(f"Loading data from {self.condfpath}...\n")
        return pd.read_csv(self.condfpath, sep='\t')

    def formatCheck(self):
        """ Ensure that condition file has the expected format. As in, 

        - `sentid`: the ID of the specific pair of sentences being compared;
                    matches what is in `predictability.tsv`
        - `sentence`: the full sentence
        - `comparison`: expected and unexpected
        - `lemma`: the word lemma
        - `pairid`: the ID of the specific pair of sentences being compared
        - `contextid`: the ID of context shared across all lemmas
        - `condition`: the specific conditions or group that the sentence
                    belongs to
        - `ROI`: the specific word positions that are of interest
        """

        NEEDS = ['sentid', 'sentence', 'lemma', 'comparison', 
                 'contextid', 'pairid', 'condition', 'ROI']

        columns = self.data.columns
        for need in NEEDS:
            assert need in columns, f"Missing {need} in {self.condfpath}"

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
                    if not LM.showSpecialTokens: 
                        continue
                    word = ''

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


    def run(self):
        """ Run MinimalPair and save result to predfpath. 

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
        for LM in self.LMs:
            outputs, word_alignments = self.gather_token_output(LM)
            self.add_entries(predictData, outputs, word_alignments, LM)

        predictData = pd.DataFrame.from_dict(predictData)
        if self.verbose:
            sys.stderr.write(f"Saving evaluations to {self.predfpath}...\n")
        predictData.to_csv(self.predfpath, index=False, sep='\t')


    def interact(self):

        word = "word" + ' '*16
        split = "Split" 
        unk = "Unk"
        model = "ModelName" + ' '*11
        surp = "surp" + ' '*4
        prob = "prob" + ' '*6
        header = f"{word} | {split} | {unk} | {model} | {surp} | {prob}"

        while True:
            sent = input('string: ').strip()
            print(header)
            print('-'*len(header))
            output = self.LMs[0].get_aligned_words_predictabilities(sent)[0]
            for word in output:
                surp = round(word.surp, 3)
                prob = round(word.prob, 5)
                print_out = f"{word.word: <20} | {word.isSplit:5} | {word.isUnk:3} | {word.modelName.split('/')[-1]: <20} | {surp: >8} | {prob: >10}"
                print(print_out)        
