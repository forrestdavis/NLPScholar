# Basic parent class for evaluations
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..utils.load_models import load_models, yield_models
import sys
import pandas as pd

class Evaluation:

    def __init__(self, config: dict, 
                **kwargs):

        # Default values
        self.predfpath = False
        self.condfpath = False
        self.loadAll = False
        self.checkFileColumns = True
        self.batchSize = 1
        self.verbose = True
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Check for paths necessary for evaluation
        if 'predfpath' in config:
            self.predfpath = config['predfpath']
        if 'condfpath' in config:
            self.condfpath = config['condfpath']

        # Set up models
        if self.loadAll:
            if self.verbose:
                sys.stderr.write('Loading all models to memory...\n')
            self.Models = load_models(config)
        else:
            self.Models = yield_models(config)
        
        self.NEEDS = None

    def columnCheck(self):
        """ Ensure that condition file has the expected columns set in
        self.NEEDS. 
        """
        columns = self.data.columns
        for need in self.NEEDS:
            assert need in columns, f"Missing {need} in {self.condfpath}"

    def load_cond(self):
        """ Load condition file 

        Returns:
            `pd.DataFrame`: pandas dataframe of data
        """
        if self.verbose:
            sys.stderr.write(f"Loading data from {self.condfpath}...\n")
        return pd.read_csv(self.condfpath, sep='\t')

    def save_evaluation(self, results: pd.DataFrame):
        """ Save evaluation results
        """
        if self.verbose:
            sys.stderr.write(f"Saving evaluations to {self.predfpath}...\n")
        results.to_csv(self.predfpath, index=False, sep='\t')

    def evaluate(self):
        """ Evaluate experiment and save result to predfpath. 

        """
        raise NotImplementedError

    def interact(self):
        """ Run interactive mode with evaluation. User is asked to input a text
        and the evaluation result is printed. 
        """
        raise NotImplementedError

