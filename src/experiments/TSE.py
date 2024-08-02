# Basic class for targeted syntactic evaluations (TSE; or minimal pair)
# experiments 
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..utils.load_models import load_models, yield_models
import pandas as pd

class TSE:

    def __init__(self, config: dict):

        # Default values
        self.saveMemory = True
        self.checkFileFormat = True
        self.batchSize = 1

        # Check for necessary things 
        assert 'predfpath' in config, "Must pass in predfpath"
        assert 'condfpath' in config, "Must pass in condfpath"
        self.predfpath = config['predfpath']
        self.condfpath = config['condfpath']

        if 'loadAll' in config:
            setattr(self, "saveMemory", False)
        if 'batchSize' in config:
            setattr(self, "batchSize", config['batchSize'])

        # Load models
        if self.saveMemory:
            self.LMs = yield_models(config)
        else:
            self.LMs = load_models(config)

        # Load data 
        self.data = self.load_cond()
        if self.checkFileFormat:
            self.formatCheck()


    def load_cond(self):
        """ Load condition file 

        Returns:
            `pd.DataFrame`: pandas dataframe of data
        """
        return pd.read_csv(self.condfpath, sep='\t')

    def formatCheck(self):
        """ Ensure that condition file has the expected format. As in, 

        - `sentid`: the ID of the specific pair of sentences being compared;
                    matches what is in `predictability.tsv`
        - `sentence`: the full sentence
        - `lemma`: the word lemma
        - `contextid`: the ID of context shared across all lemmas
        - `condition`: the specific conditions or group that the sentence
                    belongs to
        - `ROI`: the specific word positions that are of interest
        - `expected`: the comparison that is predicted to have higher
                    probability; comparison should match what is in
                    `predictability.tsv`
        """

        NEEDS = ['sentid', 'sentence', 'lemma', 'contextid', 
                 'condition', 'ROI', 'expected']

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
        conditions = self.data['condition'].tolist()

        for batch in range(len(sentences)):
            output = outputs[batch]
            sentid = sentids[batch]
            condition = conditions[batch]
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
                entriesDict['condition'].append(condition)
                entriesDict['model'].append(LM.modelname)
                entriesDict['tokenizer'].append(LM.tokenizer.tokenizername)
                entriesDict['punctuation'].append(isPunct)
                entriesDict['prob'].append(measure['probability'])
                entriesDict['surp'].append(measure['surprisal'])
        return 


    def run(self):
        """ Run TSE and save result to predfpath. 

        The output has the following information. 
        - `token`: the subword token
        - `sentid`: the ID of the specific pair of sentences being compared
        - `word`: the word the subword token belongs to
        - `wordpos`: the specific word in the context sentence that the subword token belongs to
        - `condition`: the specific condition, e.g., "grammatical" or "ungrammatical"
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
                       'condition': [], 
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
        predictData.to_csv(self.predfpath, index=False, sep='\t')


