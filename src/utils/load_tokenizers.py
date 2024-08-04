from ..tokenizers.Tokenizer import Tokenizer
from ..tokenizers.hf_tokenizer import HFTokenizer
from typing import List, Tuple, Union
from ..utils.load_kwargs import load_kwargs

tokenizer_nickname_to_model_class = {
    'hf_tokenizer': HFTokenizer,
    }

def get_tokenizer_instance(tokenizer_type: str, config: dict
                      ) -> Tuple[Union[Tokenizer], dict]:
    """ Returns the relevant tokenizer class and sets the kwargs from config.

    Args:
        tokenizer_type (`str`): Nickname of model class
        config (`dict`): config file with model and tokenizer fields (see
                            README)

    Returns:
        `Tuple[Tokenizer, dict]`: Tuple of Tokenizer class and dictionary with
                                    optional parameters. 

    """
    tokenizer_cls = tokenizer_nickname_to_model_class[tokenizer_type]
    assert issubclass(tokenizer_cls, Tokenizer)
    kwargs = load_kwargs(config)

    return tokenizer_cls, kwargs

def load_tokenizers(config: dict) -> List[Tokenizer]:
    """ Loads instances of tokenizers specified in config.

    Args:
        config (`dict`): config file with tokenizer fields (see README)
    Returns:
        `List[Tokenizer]`: List of Tokenizer instances
    """
    return_tokenizers = []
    for tokenizer_type in config['tokenizers']:
        if tokenizer_type not in tokenizer_nickname_to_model_class:
            raise ValueError(f"Unrecognized tokenizer: {tokenizer_type}")
        for tokenizer_instance in config['tokenizers'][tokenizer_type]:
            tokenizer_cls, kwargs = get_tokenizer_instance(tokenizer_type, config)
            return_tokenizers.append(tokenizer_cls(tokenizer_instance, **kwargs))
    return return_tokenizers
    
