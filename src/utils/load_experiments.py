from ..experiments.Interact import Interact
from ..experiments.TSE import TSE
from ..experiments.TokenClassification import TokenClassification
from ..experiments.TextClassification import TextClassification

experiment_nickname_to_experiment_class = {
    'TSE': TSE, 
    'Interact': Interact, 
    'TokenClassification': TokenClassification, 
    'TextClassification': TextClassification,
    }

def load_experiment(config: dict):
    """ Loads instances of experiment specified in config.

    Args:
        config (`dict`): config file with tokenizer fields (see README)
    Returns:
        `Union[Interact, TSE, TokenClassification, TextClassification]`:
            Experiment Instance
    """
    experiment_type = config['exp']
    if experiment_type not in experiment_nickname_to_experiment_class:
        raise ValueError(f"Unrecognized experiment: {experiment_type}")
    experiment_cls = experiment_nickname_to_experiment_class[experiment_type]
    return experiment_cls(config)
