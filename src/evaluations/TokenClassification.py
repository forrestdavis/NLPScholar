# Basic class for token classification experiments
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..utils.load_models import load_models, yield_models
import pandas as pd

class TokenClassification:

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
        - `target`: the target labels (space separated) that this text belongs to
                    Note: this assume you have one per word, if you are missing
                    words, the labels will be aligned left-to-right with None as
                    the padded target. 
        """

        NEEDS = ['textid', 'text', 'condition', 'target']

        columns = self.data.columns
        for need in NEEDS:
            assert need in columns, f"Missing {need} in {self.condfpath}"

    def gather_token_output(self, Classifier):
        """ Returns the outputs and word alignments for a classification model
        for the experiment. 

        Args:
            Classifier (`Classifier`): An instance of a Classifier

        Returns:
            `list`: List of lists for each batch comprised of dictionaries per
                    token. Each dictionary has token_id, label, probability, and
                    id.  Note: the padded output is included.
            `List[dict]`: List with a dictionary for each batch. Dictionary has
                        two elements: mapping_to_words, which is a list with the
                        word number each token belongs to, and words, which is a
                        list of each token's word string. 

        """
        sentences = self.data['text'].tolist()
        outputs = []
        word_alignments = []
        for batch_idx in range(0, len(sentences), self.batchSize):
            batch = sentences[batch_idx:batch_idx+self.batchSize]
            outputs.extend(Classifier.get_by_token_predictions(batch))
            word_alignments.extend(Classifier.tokenizer.align_words_ids(batch))
        return outputs, word_alignments

    def add_entries(self, entriesDict, outputs, word_alignments, Classifier):
        """ Adds entries to the dictionary of results.
        """
        sentences = self.data['text'].tolist()
        sentids = self.data['textid'].tolist()
        conditions = self.data['condition'].tolist()
        all_targets = self.data['target'].tolist()

        for batch in range(len(sentences)):
            output = outputs[batch]
            sentid = sentids[batch]
            condition = conditions[batch]
            alignments = word_alignments[batch]['mapping_to_words']
            words = word_alignments[batch]['words']
            targets = all_targets[batch].split()

            for measure, alignment, word in zip(output, alignments, words,
                                                strict=True):
                
                target = None

                # Skip pad tokens 
                if Classifier.tokenizer.pad_token_id == measure['token_id']:
                    continue

                # Get token
                token = Classifier.tokenizer.convert_ids_to_tokens(measure['token_id'])

                if word is None:
                    if not Classifier.showSpecialTokens: 
                        continue
                    word = ''

                if alignment is not None and alignment < len(targets):
                    target = targets[alignment]
                # Figure out if punctuation 
                isPunct = False
                if Classifier.tokenizer.TokenIDIsPunct(measure['token_id']):
                    isPunct = True

                # Add to prediction data
                entriesDict['token'].append(token)
                entriesDict['textid'].append(sentid)
                entriesDict['word'].append(word)
                entriesDict['wordpos'].append(alignment)
                entriesDict['condition'].append(condition)
                entriesDict['model'].append(Classifier.modelname)
                entriesDict['tokenizer'].append(Classifier.tokenizer.tokenizername)
                entriesDict['punctuation'].append(isPunct)
                entriesDict['target'].append(target)
                entriesDict['predicted'].append(measure['label'])
                entriesDict['prob'].append(measure['probability'])
        return 


    def run(self):
        """ Run token classification and save result to predfpath. 

        The output has the following information. 
        - `token`: the subword token
        - `textid`: the ID of the specific pair of text being compared;
                    matches what is in `predictability.tsv`
        - `word`: the word the subword token belongs to
        - `wordpos`: the specific word in the context sentence that the subword token belongs to
        - `condition`: the specific condition, e.g., "grammatical" or "ungrammatical"
        - `model`: the name of the classifier
        - `tokenizer`: the name of the tokenizer
        - `punctuation`: is the token punctuation?
        - `target`: the target label that this text belongs to (or None if none
                    was specified)
        - `predicted`: the predicted label for this text belong
        - `prob`: the probability of the token given the context

        """
        predictData = {'token': [], 
                       'textid': [], 
                       'word': [],
                       'wordpos': [], 
                       'condition': [], 
                       'model': [],
                       'tokenizer': [],
                       'punctuation': [],
                       'target': [],
                       'predicted': [],
                       'prob': [], 
                      }
        for Classifier in self.Classifiers:
            outputs, word_alignments = self.gather_token_output(Classifier)
            self.add_entries(predictData, outputs, word_alignments, Classifier)

        predictData = pd.DataFrame.from_dict(predictData)
        predictData.to_csv(self.predfpath, index=False, sep='\t')


