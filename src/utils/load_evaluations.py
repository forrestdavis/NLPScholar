from ..evaluations.MinimalPair import MinimalPair
from ..evaluations.TokenClassification import TokenClassification
from ..evaluations.TextClassification import TextClassification
from ..evaluations.LanguageModel import LanguageModel
from ..evaluations.WordPredictability import WordPredictability
from .load_kwargs import load_kwargs

evaluation_nickname_to_evaluation_class = {
    'MinimalPair': MinimalPair, 
    'TokenClassification': TokenClassification, 
    'TextClassification': TextClassification,
    'LanguageModel': LanguageModel,
    'WordPredictability': WordPredictability
    }

def load_evaluation(config: dict):
    """ Loads instances of evaluation specified in config.

    Args:
        config (`dict`): config file with tokenizer fields (see README)
    Returns:
        `Union[MinimalPair, TokenClassification, TextClassification]`:
            Evaluation Instance
    """
    evaluation_type = config['exp']
    if evaluation_type not in evaluation_nickname_to_evaluation_class:
        raise ValueError(f"Unrecognized evaluation: {evaluation_type}")
    kwargs = load_kwargs(config)
    evaluation_cls = evaluation_nickname_to_evaluation_class[evaluation_type]
    return evaluation_cls(config, **kwargs)
