from .Classifier import Classifier
from .hf_text_classification_model import HFTextClassificationModel
from typing import List

model_nickname_to_model_class = {
    'hf_text_classification_model': HFTextClassificationModel,
    }

def get_model_instance(model_type: str, config: dict): 
    """ Returns the relevant model class and sets the kwargs from config.

    Args:
        model_type (`str`): Nickname of model class
        config (`dict`): config file with model and tokenizer fields (see README)

    Returns:
        `Tuple[Classifier, dict]`: Tuple of Classifier class and dictionary with
                                    optional parameters. 

    """
    model_cls = model_nickname_to_model_class[model_type]
    assert issubclass(model_cls, Classifier)

    kwargs = {}
    if 'precision' in config:
        kwargs['precision'] = config['precision']
    if 'device' in config:
        kwargs['device'] = config['device']

    return model_cls, kwargs

def create_tokenizer_configs(config: dict) -> List[dict]:
    """ Returns list of individual tokenizer config dict. 

    Args:
        config (`dict`): config file with model and tokenizer fields (see README)

    Returns:
        `List[dict]`: List of tokenizer dicts
    """
    tokenizer_configs = []
    if 'tokenizers' in config:
        for tokenizer_type in config['tokenizers']:
            for tokenizer_instance in config['tokenizers'][tokenizer_type]:
                tokenizer_configs.append({'tokenizers': {tokenizer_type:
                                                     [tokenizer_instance]}})
    return tokenizer_configs

def load_classifiers(config: dict) -> List[Classifier]:
    """ Loads instances of classifiers specified in config.

    Args:
        config (`dict`): config file with model and tokenizer fields (see README)

    Returns:
        `List[Classifier]`: List of model instances 
    """
    return_models = []
    tokenizer_configs = create_tokenizer_configs(config)
    for model_type in config['models']:
        if model_type not in model_nickname_to_model_class:
            raise ValueError(f"Unrecognized model: {model_type}")
        for model_instance in config['models'][model_type]:
            model_cls, kwargs = get_model_instance(model_type, config)
            if tokenizer_configs:
                tokenizer_config = tokenizer_configs.pop(0)
            else:
                tokenizer_config = None
            return_models.append(model_cls(model_instance,
                                           tokenizer_config, 
                                               **kwargs))
    return return_models
    
def yield_classifiers(config: dict):
    """ Yield instances of models specified in config.

    Args:
        config (`dict`): config file with model and tokenizer fields (see README)

    Yields:
        `Classifier`: model instances 
    """
    tokenizer_configs = create_tokenizer_configs(config)
    for model_type in config['models']:
        if model_type not in model_nickname_to_model_class:
            raise ValueError(f"Unrecognized model: {model_type}")
        for model_instance in config['models'][model_type]:
            model_cls, kwargs = get_model_instance(model_type, config)
            if tokenizer_configs:
                tokenizer_config = tokenizer_configs.pop(0)
            else:
                tokenizer_config = None
            yield model_cls(model_instance, tokenizer_config, **kwargs)
