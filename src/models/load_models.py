from .LM import LM
from .hf_causal_model import HFCausalModel
from .hf_masked_model import HFMaskedModel
from typing import List

model_nickname_to_model_class = {
    'hf_causal_model': HFCausalModel,
    'hf_masked_model': HFMaskedModel,
    }

def load_models(config: dict) -> List[LM]:
    """ Loads instances of models specified in config.

    Args:
        config (`dict`): config file with model and tokenizer fields (see README)
    Returns:
        `List[LM]`: List of model instances
    """
    return_models = []
    for model_type in config['models']:
        if model_type not in model_nickname_to_model_class:
            raise ValueError(f"Unrecognized model: {model_type}")
        for model_instance in config['models'][model_type]:
            model_cls = model_nickname_to_model_class[model_type]
            assert issubclass(model_cls, LM)

            kwargs = {}
            if 'notOffset' in config:
                kwargs['offset'] = False
            if 'getHidden' in config:
                kwargs['getHidden'] = True
            if 'precision' in config:
                kwargs['precision'] = config['precision']
            if 'ignorePunct' in config:
                kwargs['includePunct'] = False
            if 'device' in config:
                kwargs['device'] = config['device']
            return_models.append(model_cls(model_instance, config, 
                                           **kwargs))
    return return_models
    
