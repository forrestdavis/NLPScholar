from ..analysis.MinimalPair import MinimalPair
from ..analysis.TokenClassification import TokenClassification
from ..analysis.TextClassification import TextClassification
from .load_kwargs import load_kwargs

analysis_nickname_to_analysis_class = {
    'MinimalPair': MinimalPair, 
    'TokenClassification': TokenClassification, 
    'TextClassification': TextClassification,
    }

def load_analysis(config: dict):
    """ Loads instances of analysis specified in config.

    Args:
        config (`dict`): config file with tokenizer fields (see README)
    Returns:
        `Union[MinimalPair, TokenClassification, TextClassification]`:
            Analysis Instance
    """
    analysis_type = config['exp']
    if analysis_type not in analysis_nickname_to_analysis_class:
        raise ValueError(f"Unrecognized analysis: {analysis_type}")
    kwargs = load_kwargs(config)
    analysis_cls = analysis_nickname_to_analysis_class[analysis_type]
    return analysis_cls(config, **kwargs)
