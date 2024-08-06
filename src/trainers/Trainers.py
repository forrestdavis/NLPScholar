# Basic trainer parent class
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..utils.load_models import load_models, yield_models
import torch
from typing import Union, Dict, List, Tuple, Optional
import sys
import datasets
import pandas as pd

class Trainer:

    def __init__(self, config: dict, 
                **kwargs):

        # Default values
        self.trainfpath = None
        self.validfpath = None
        self.batchSize = 1
        self.verbose = True
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Set up model
        self.Model = load_models(config)
        assert len(self.Model) == 1, "You can only train one model at a time"
        self.Model = self.Model[0]

        # Set up the raw dataset 
        self.set_dataset()

        self.data_collator = None
        self.evaluator = None

    def load_train_valid(self) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """ Load the training and optional validation data as hf datasets. 
        
        Note: 
            Looks for self.trainfpath which must be specified and optionally
            looks for self.validfpath
        """
        assert self.trainfpath is not None, "Must pass a trainfpath" \
                                " for training"
        valid = None
        if ':' in self.trainfpath: 
            path, split = self.trainfpath.split(':')
            train = self.load_from_hf_dataset(path, split)
        else:
            train = self.load_from_tsv(self.trainfpath)

        if self.validfpath is not None:
            if ':' in self.validfpath:
                path, split = self.validfpath.split(":")
                valid = self.load_from_hf_dataset(path, split)
            else:
                valid = self.load_from_tsv(path)
        return train, valid
        
    def load_from_hf_dataset(self, path: str, split: str
                            ) -> datasets.Dataset:
        """ Returns a split of a hf dataset. 

        Args:
            path (`str`): Path of dataset
            split (`str`): Name of split 
        
        Returns:
            `datasets.Dataset`: Dataset instance
        """
        return datasets.load_dataset(path, split=split)

    def load_from_tsv(self, path: str) -> datasets.Dataset:
        """ Returns a hf dataset from a tsv file. 

        Args:
            path (`str`): Path of dataset

        Returns:
            `dataset.Dataset`: Dataset instance
        """
        return datasets.Dataset.from_pandas(pd.read_csv(path, sep='\t'))

    def set_dataset(self):
        """ Sets the dataset attribute with a dataset dict. 
        """
        if self.verbose:
            sys.stderr.write('Loading the dataset...\n')
        train, valid = self.load_train_valid()
        self.dataset = datasets.DatasetDict({'train': train, 'valid': valid})

    def preprocess_dataset(self):
        """ Tokenize the input as necessary for the task. This should update the
        self. dataset attribute and the self.data_collator attribute (if
        applicable). This should be called in the __init__ of the child class. 
        """
        raise NotImplementedError

    def compute_metrics(self, prediction):
        pass

    def train(self):
        pass 
