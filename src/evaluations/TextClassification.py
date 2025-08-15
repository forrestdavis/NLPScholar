# Basic class for text classification experiments
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from .Evaluation import Evaluation
import pandas as pd

class TextClassification(Evaluation):

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)
        self.NEEDS = ['textid', 'text', 'target']

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
            batch_pair = None
            if pair is not None:
                batch_pair = pair[batch_idx:batch_idx+self.batchSize]
            outputs.extend(Classifier.get_text_predictions(batch, batch_pair))
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
            entriesDict['predicted'].append(output['predicted label'])
            entriesDict['prob'].append(output['probability'])
            if self.giveAllLabels:
                for entry in output['all labels']:
                    label, prob = entry['label'], entry['probability']
                    if label not in entriesDict:
                        entriesDict[label] = []
                    entriesDict[label].append(prob)
        return 


    def evaluate(self):
        """ Run text classification and save result to predfpath. 

        The output has the following information. 
        - `textid`: the ID of the specific pair of text being compared;
                    matches what is in `predictability.tsv`
        - `target`: the target label that this text belongs to 
        - `model`: the name of the classifier
        - `tokenizer`: the name of the tokenizer
        - `predicted`: the predicted label for this text belong
        - `prob`: the predicted label's probability
        #TODO: Add details about other labels
        """
        # Load data 
        assert self.datafpath is not None, "Missing a datafpath value\n"
        assert self.predfpath is not None, "Missing a predfpath value\n"

        self.data = self.load_cond()
        if self.checkFileColumns:
            self.columnCheck()

        predictData = {'textid': [], 
                       'target': [], 
                       'model': [], 
                       'tokenizer': [],
                       'predicted': [], 
                       'prob': [],
                      }
        for Classifier in self.Models:
            outputs = self.gather_labeled_output(Classifier)
            self.add_entries(predictData, outputs, Classifier)

        predictData = pd.DataFrame.from_dict(predictData)
        self.save_evaluation(predictData)

    def interact(self):
        """ Run interactive mode with TextClassification. User is asked to input
        a text and an optional pair (for tasks like natural language inference).

        Prints:
            `text`: The text inputted (shortened to fit on screen)
            `ModelName`: Name of the model (shortened to fit on screen)
            `label`: The most likely classification label 
            `prob`: Probability of the label of the text 
        """

        text = "text" + ' '*20
        model = "ModelName" + ' '*16
        label = "label" + ' '*6
        prob = "prob" + ' '*6
        header = f"{text} | {model} | {label} | {prob}"

        # Only consider the first model
        if self.loadAll:
            Classifier = self.Models[0]
        else:
            Classifier = next(self.Models)

        while True:
            text = input('text (or STOP): ').strip()
            if text == 'STOP':
                break
            pair = input('pair (optional): ').strip()
            print(header)
            print('-'*len(header))
            if pair == '':
                pair = None
            output = Classifier.get_text_predictions(text, pair)[0]
            if len(text) > 24:
                text = text[:21] + '...'
            modelName = str(Classifier)
            if len(modelName) > 24:
                modelName = modelName[:21] + '...'
            prob = round(output['probability'], 5)
            print_out = f"{text: <24} | {modelName: <24}"\
                        f" | {output['label']: <11} | {prob: >10}"
            print(print_out)
