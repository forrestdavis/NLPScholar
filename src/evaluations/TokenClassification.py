# Basic class for token classification experiments
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .Evaluation import Evaluation
import pandas as pd

class TokenClassification(Evaluation):

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)
        self.NEEDS = ['textid', 'text', 'condition', 'target']

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


    def evaluate(self):
        """ Evaluate TokenClassification and save result to predfpath. 

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
        # Load data 
        assert self.datafpath is not None, "Missing a datafpath value\n"
        assert self.predfpath is not None, "Missing a predfpath value\n"

        self.data = self.load_cond()
        if self.checkFileColumns:
            self.columnCheck()

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
        for Classifier in self.Models:
            outputs, word_alignments = self.gather_token_output(Classifier)
            self.add_entries(predictData, outputs, word_alignments, Classifier)

        predictData = pd.DataFrame.from_dict(predictData)
        self.save_evaluation(predictData)

    def interact(self):
        """ Run interactive mode with MinimalPair. User is asked to input a text
        and the by-word predictability measures are given.

        Prints:
            `token`: The word given by tokenizer 
            `ModelName`: Name of the model (shortened to fit on screen)
            `label`: The most likely classification label 
            `prob`: Probability of the label of the token 
        """

        token = "token" + ' '*15
        model = "ModelName" + ' '*16
        label = "label" + ' '*6
        prob = "prob" + ' '*6
        header = f"{token} | {model} | {label} | {prob}"

        # Only consider the first model
        if self.loadAll:
            Classifier = self.Models[0]
        else:
            Classifier = next(self.Models)

        modelName = str(Classifier)
        if len(modelName) > 24:
            modelName = modelName[:21] + '...'

        while True:
            sent = input('text (or STOP): ').strip()
            if sent == 'STOP':
                break
            print(header)
            print('-'*len(header))
            output = Classifier.get_by_token_predictions(sent)[0]
            word_alignment = \
                    Classifier.tokenizer.align_words_ids(sent)[0]['words']
            for measure, unit in zip(output, word_alignment, strict=True):

                token = \
                    Classifier.tokenizer.convert_ids_to_tokens(
                        measure['token_id'])
                prob = round(measure['probability'], 5)
                
                label = measure['label']
                print_out = f"{token: <20} | {modelName: <24} | {label: <11}" \
                f" | {prob: >10}"
                if unit is None:
                    if Classifier.showSpecialTokens:
                        print(print_out)
                    continue
                print(print_out)
