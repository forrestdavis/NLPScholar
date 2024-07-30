from .Tokenizer import Tokenizer
from .hf_tokenizer import HFTokenizer
from typing import List

tokenizer_nickname_to_model_class = {
    'hf': HFTokenizer,
    }

def load_tokenizers(config: dict) -> List[Tokenizer]:
    """ Loads instances of tokenizers specified in config.

    Args:
        config (`dict`): config file with tokenizer fields (see README)
    Returns:
        `List[Tokenizer]`: List of Tokenizer instances
    """
    return_tokenizers = []
    for tokenizer_type in config['tokenizer']:
        if tokenizer_type not in tokenizer_nickname_to_model_class:
            raise ValueError(f"Unrecognized tokenizer: {tokenizer_type}")
        for tokenizer_instance in config['tokenizer'][tokenizer_type]:
            tokenizer_cls = tokenizer_nickname_to_model_class[tokenizer_type]
            assert issubclass(tokenizer_cls, Tokenizer)
            return_tokenizers.append(tokenizer_cls(tokenizer_instance))
    return return_tokenizers
    
