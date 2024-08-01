# Basic class for targeted syntactic evaluations (TSE; or minimal pair)
# experiments 
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..models.load_models import load_models, yield_models
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

        - `sentid` (the ID of the specific pair of sentences being compared;
                    matches what is in `predictability.tsv`)
        - `sentence` (the full sentence)
        - `lemma` (the word lemma)
        - `contextid` (the ID of context shared across all lemmas)
        - `condition` (the specific conditions or group that the sentence
                    belongs to)
        - `ROI` (the specific word positions that are of interest)
        - `expected` (the comparison that is predicted to have higher
                    probability; comparison should match what is in
                    `predictability.tsv`)
        """

        NEEDS = ['sentid', 'sentence', 'lemma', 'contextid', 
                 'condition', 'ROI', 'expected']

        columns = self.data.columns
        for need in NEEDS:
            assert need in columns, f"Missing {need} in {self.condfpath}"

    def gather_token_output(self, LM):
        sentences = self.data['sentence'].tolist()
        outputs = []
        for batch_idx in range(0, len(sentences), self.batchSize):
            batch = sentences[batch_idx:batch_idx+self.batchSize]
            outputs.extend(LM.get_by_token_predictability(batch))
        return outputs

    def add_entries(self, entriesDict, outputs, LM):
        sentences = self.data['sentence'].tolist()
        sentids = self.data['sentid'].tolist()
        conditions = self.data['condition'].tolist()
        word_alignments = LM.tokenizer.align_words_ids(sentences, LM.language)

        for idx in range(len(sentences)):
            words, alignments = word_alignments[idx]
            output = outputs[idx]
            sentid = sentids[idx]
            condition = conditions[idx]

            for word_idx, (word, alignment) in enumerate(zip(words,
                                                             alignments)):
                token_predict = output.pop(0)
                word_token_id = alignment.pop(0)
                # Handle non-surface tokens (e.g., [CLS])
                while token_predict['input_id'] != word_token_id:
                    token_predict = output.pop(0)

                keepAligning = True
                while keepAligning: 
                    assert token_predict['input_id'] == word_token_id
                    isPunct = 0
                    if LM.tokenizer.TokenIDIsPunct(word_token_id):
                        isPunct = 1

                    token = LM.tokenizer.convert_ids_to_tokens(word_token_id)

                    # Add to prediction data
                    entriesDict['token'].append(token)
                    entriesDict['sentid'].append(sentid)
                    entriesDict['wordpos'].append(word_idx)
                    entriesDict['condition'].append(condition)
                    entriesDict['model'].append(LM.modelname)
                    entriesDict['tokenizer'].append(LM.tokenizer.tokenizername)
                    entriesDict['punctuation'].append(isPunct)
                    entriesDict['prob'].append(token_predict['probability'])
                    entriesDict['surp'].append(token_predict['surprisal'])
                    
                    if alignment:
                        token_predict = output.pop(0)
                        word_token_id = alignment.pop(0)
                    else:
                        keepAligning = False

    def run(self):
        predictData = {'token': [], 
                       'sentid': [], 
                       'wordpos': [], 
                       'condition': [], 
                       'model': [],
                       'tokenizer': [],
                       'punctuation': [],
                       'prob': [], 
                       'surp': [],
                      }
        for LM in self.LMs:
            outputs = self.gather_token_output(LM)
            self.add_entries(predictData, outputs, LM)

        predictData = pd.DataFrame.from_dict(predictData)
        print(predictData)


