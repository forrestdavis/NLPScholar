from ..models.LM import LM
from ..models.hf_causal_model import HFCausalModel
from ..models.hf_masked_model import HFMaskedModel

from ..classifiers.Classifier import Classifier
from ..classifiers.hf_text_classification_model import HFTextClassificationModel
from ..classifiers.hf_token_classification_model import HFTokenClassificationModel

from ..utils.load_kwargs import load_kwargs

from typing import List, Union, Tuple

model_nickname_to_model_class = {
    'hf_causal_model': HFCausalModel,
    'hf_masked_model': HFMaskedModel,
    'hf_text_classification_model': HFTextClassificationModel, 
    'hf_token_classification_model': HFTokenClassificationModel,
    }

def get_model_instance(model_type: str, config: dict
                      ) -> Tuple[Union[LM,Classifier], dict]:
    """ Returns the relevant model class and sets the kwargs from config.

    Args:
        model_type (`str`): Nickname of model class
        config (`dict`): config file with model and tokenizer fields (see README)

    Returns:
        `Tuple[Union[LM, Classifier], dict]`: Tuple of LM or Classifier class
                                and dictionary with optional parameters. 

    """
    model_cls = model_nickname_to_model_class[model_type]
    assert issubclass(model_cls, LM) or issubclass(model_cls, Classifier)
    kwargs = load_kwargs(config)

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

def load_models(config: dict) -> List[Union[LM, Classifier]]:
    """ Loads instances of models specified in config.

    Args:
        config (`dict`): config file with model and tokenizer fields (see README)

    Returns:
        `List[Union[LM, Classifier]]`: List of model or classifier instances 
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
    
def yield_models(config: dict):
    """ Yield instances of models specified in config.

    Args:
        config (`dict`): config file with model and tokenizer fields (see README)

    Yields:
        `Union[LM, Classifier]`: model or classifier instances 
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
