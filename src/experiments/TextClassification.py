# Basic class for text classification experiments
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..utils.load_models import load_models, yield_models
import pandas as pd

class TextClassification:

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
            self.Classifiers = yield_models(config)
        else:
            self.Classifiers = load_models(config)

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

        - `textid`: the ID of the specific pair of text being compared;
                    matches what is in `predictability.tsv`
        - `text`: the full text
        - `pair`: *optional* second sentence to use as a pair 
        - `condition`: the specific conditions or group that the text
                    belongs to
        - `target`: the target label that this text belongs to 
        """

        NEEDS = ['textid', 'text', 'condition', 'target']

        columns = self.data.columns
        for need in NEEDS:
            assert need in columns, f"Missing {need} in {self.condfpath}"

    def gather_labeled_output(self, Classifier):
        """ Returns the outputs and word alignments for a langaguage model for
            the experiment. 

        Args:
            Classifier (`Classifier`): An instance of a Classifier

        Returns:
            `list`: List of lists for each batch comprised of dictionaries per
                    token. Each dictionary has token_id, probability, surprisal.
                    Note: the padded output is included.
            `List[dict]`: List with a dictionary for each batch. Dictionary has
                        two elements: mapping_to_words, which is a list with the
                        word number each token belongs to, and words, which is a
                        list of each token's word string. 

        """
        text = self.data['text'].tolist()
        pair = None
        if 'pair' in self.data.columns:
            pair = self.data['pair'].tolist()
        outputs = []
        for batch_idx in range(0, len(text), self.batchSize):
            batch = text[batch_idx:batch_idx+self.batchSize]
            if pair is not None:
                batch_pair = pair[batch_idx:batch_idx+self.batchSize]
            outputs.extend(Classifier.get_text_predictions(batch, pair))
        return outputs

    def add_entries(self, entriesDict, outputs, Classifier):
        """ Adds entries to the dictionary of results.
        """
        textids = self.data['textid'].tolist()
        targets = self.data['target'].tolist()

        for textid, target, output in zip(textids, targets, outputs,
                                          strict=True):

            entriesDict['textid'].append(textid)
            entriesDict['target'].append(target)
            entriesDict['model'].append(Classifier.modelname)
            entriesDict['tokenizer'].append(Classifier.tokenizer.tokenizername)
            entriesDict['predicted'].append(output['label'])
            entriesDict['prob'].append(output['probability'])

        return 


    def run(self):
        """ Run TSE and save result to predfpath. 

        The output has the following information. 
        - `textid`: the ID of the specific pair of text being compared;
                    matches what is in `predictability.tsv`
        - `target`: the target label that this text belongs to 
        - `model`: the name of the classifier
        - `tokenizer`: the name of the tokenizer
        - `predicted`: the predicted label for this text belong
        - `prob`: the predicted label's probability
        """
        predictData = {'textid': [], 
                       'target': [], 
                       'model': [], 
                       'tokenizer': [],
                       'predicted': [], 
                       'prob': [],
                      }
        for Classifier in self.Classifiers:
            outputs = self.gather_labeled_output(Classifier)
            self.add_entries(predictData, outputs, Classifier)

        predictData = pd.DataFrame.from_dict(predictData)
        predictData.to_csv(self.predfpath, index=False, sep='\t')


