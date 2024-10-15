# Basic class for language modeling evaluations
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# October 2024

from .Evaluation import Evaluation
import pandas as pd

class LanguageModel(Evaluation):

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)

    def evaluate(self):
        raise NotImplementedError

    def interact(self):
        """ Run interactive mode with LanguageModel. User is asked to input a text
        and the perpelxity is returned.

        Prints:
            `ModelName`: Name of the model (shortened to fit on screen)
            `perplexity`: Perplexity of the text
        """

        model = "ModelName" + ' '*11
        ppl = "perplexity" + ' '*6
        header = f"{model} | {ppl}"

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
            ppl = LM.get_by_batch_perplexity(sent)['perplexity'][0]
            ppl = round(ppl, 10)
            print_out = f"{LM.modelname.split('/')[-1]: <20} | "\
                        f"{ppl: >16}"
            print(print_out)
