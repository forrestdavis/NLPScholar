# Basic trainer parent class
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..utils.load_models import load_models
import torch 
from typing import Union, Dict, List, Tuple, Optional
import sys
import datasets
import random

class Trainer:

    def __init__(self, config: dict, 
                **kwargs):

        # Default values
        self.trainfpath = None
        self.validfpath = None
        self.verbose = True

        # Dataset defaults
        self.seed = 23
        self.samplePercent = 10
        self.textLabel = 'text'
        self.pairLabel = 'pair'
        self.tokensLabel = 'tokens'
        self.tagLabel = 'tags'
        self.dataset = None

        # Training defaults
        self.modelfpath = None
        self.epochs = 2
        self.eval_strategy = 'epoch'
        self.eval_steps = 500
        self.batchSize = 8
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.save_strategy = 'epoch'
        self.save_steps = 500
        self.load_best_model_at_end = False
        self.wholeWordMasking = False
        self.maskProbability = 0.15
        self.maxSequenceLength = 128

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Set up model
        config = {**config, **kwargs}
        self.Model = load_models(config)
        assert len(self.Model) == 1, "You can only train one model at a time"
        self.Model = self.Model[0]

        assert self.modelfpath is not None, "Must pass a modelfpath" \
                                    " for saving the final model"

        self.data_collator = None
        self.evaluator = None

    def load_train_valid(self) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """ Load the training and validation data as hf datasets. 
        
        Note: 
            Looks for self.trainfpath and self.validfpath which must be
            specified 
        """
        assert self.trainfpath is not None, "Must pass a trainfpath" \
                                " for training"
        assert self.validfpath is not None, "Must pass a validfpath" \
                                " for validation during training"

        def get_info(name: str) -> tuple:
            info = name.split(':')
            if len(info) == 3:
                path, task, split = info
            else:
                task = None
                path, split = info
            return path, task, split

            
        if ':' in self.trainfpath: 
            path, task, split = get_info(self.trainfpath)
            train = self.load_from_hf_dataset(path, task, split)
        else:
            train = self.load_from_tsv(self.trainfpath)

        if ':' in self.validfpath:
            path, task, split = get_info(self.validfpath)
            valid = self.load_from_hf_dataset(path, task, split)
        else:
            valid = self.load_from_tsv(self.validfpath)

        # Shuffle dataset, which is necessary if we are selecting a subset
        # (e.g., what if all the 1 labels occur in the last 20% of the data?
        train = train.shuffle(seed = self.seed)
        valid = valid.shuffle(seed = self.seed)

        if self.samplePercent is not None:
            if self.samplePercent >= 1:
                self.samplePercent = self.samplePercent/100
            train = train.select(range(int(self.samplePercent*len(train))))
            valid = valid.select(range(int(self.samplePercent*len(valid))))

        return train, valid
        
    def load_from_hf_dataset(self, path: str, task: str, split: str
                            ) -> datasets.Dataset:
        """ Returns a split of a hf dataset. 

        Args:
            path (`str`): Path of dataset
            task (`str`): Task (or name) of sub-dataset (e.g., "mnli" for glue)
            split (`str`): Name of split 
        
        Returns:
            `datasets.Dataset`: Dataset instance
        """
        return datasets.load_dataset(path, name=task, split=split)

    def load_from_tsv(self, path: str) -> datasets.Dataset:
        """ Returns a hf dataset from a tsv file. 

        Args:
            path (`str`): Path of dataset

        Returns:
            `dataset.Dataset`: Dataset instance
        """
        return datasets.load_dataset('csv', data_files=path, delimiter="\t",
                                     split='train')

    def load_from_json(self, path: str) -> datasets.Dataset:
        """ Returns a hf dataset from a json file. 

        Args:
            path (`str`): Path of dataset

        Returns:
            `dataset.Dataset`: Dataset instance
        """
        return datasets.load_dataset('json', data_files=path,
                                     split='train')

    def set_dataset(self):
        """ Sets the dataset attribute with a dataset dict. 
        """
        if self.verbose:
            sys.stderr.write('Loading the dataset...\n')
        train, valid = self.load_train_valid()
        self.dataset = datasets.DatasetDict({'train': train, 'valid':
                                             valid})

    def show_k_samples(self, k: int = 5): 
        """ Print k samples from the training data. 

        Args: 
            k (`int`): Number of samples to show
        """
        # Get k random indices
        idxs = []
        assert k < len(self.dataset['train']), f"There are less than {k}"\
                                            " samples"
        random_ints = random.sample(range(len(self.dataset['train'])), k=k)
        for idx in random_ints:
            print(self.dataset['train'].select([idx]))

    def preprocess_dataset(self):
        """ Tokenize the input as necessary for the task. This should update the
        self.dataset attribute and the self.data_collator attribute (if
        applicable). This should be called in the train function of the child
        class. 
        """
        raise NotImplementedError

    def compute_metrics(self, prediction):
        """ Computes performance metrics on predictions from the model on the
        validation data during training. Note: Careful about the version of
        transformers that you have!
        """
        raise NotImplementedError

    def train(self):
        """ Trains the model on the task using the hyperparameters specified in
        config (or the defaults set in __init__). The options are lifted (with
        only slight name changes occasionally) from HuggingFace's Trainer
        arguments and Trainer class:
            https://huggingface.co/docs/transformers/main_classes/trainer#trainer

            `self.modelfpath`: The output directory where the model is saved
            `self.epochs`: Number of training epochs. The default is 2. 
            `self.eval_strategy`: The evaluation strategy to use ('no' | 'steps'
                            | 'epoch'). The default is 'epoch'.
            `self.eval_steps`: Number of update steps between two evaluations.
                                The default is 500. 
            `self.batchSize`: The per-device batch size for train/eval. The
                                default is 8. 
            `self.learning_rate`: The initial learning rate for AdamW. Default
                                    is 5e-5. 
            `self.weight_decay`: Weight decay. The default is 0.01.
            `self.save_strategy`: The checkpoint save strategy to use ('no'
                            | 'steps' | 'epoch'). The default is 'epoch'.
            `self.save_steps`: Number of update steps between saves. Default is
                                500.
            `self.load_best_model_at_end`: Whether or not to load the best model
                            at the end of training. If True, the best model will
                            be saved as the model at the end. Default is False.
        """
        raise NotImplementedError
