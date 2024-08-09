from ..trainers.Trainer import Trainer
from ..trainers.HFLanguageModelTrainer import HFLanguageModelTrainer
from ..trainers.HFTextClassificationTrainer import HFTextClassificationTrainer
from ..trainers.HFTokenClassificationTrainer import HFTokenClassificationTrainer
from .load_kwargs import load_kwargs

trainer_nickname_to_trainer_class = {
    'MinimalPair': HFLanguageModelTrainer, 
    'TokenClassification': HFTokenClassificationTrainer, 
    'TextClassification': HFTextClassificationTrainer,
    }

def load_trainer(config: dict):
    """ Loads instances of trainer specified in config.

    Args:
        config (`dict`): config file with tokenizer fields (see README)
    Returns:
        `Union[Trainer, HFLanguageModelTrainer, HFTextClassificationTrainer,
        HFToHFTokenClassificationTrainer]`: Trainer Instance
    """
    trainer_type = config['exp']
    if trainer_type not in trainer_nickname_to_trainer_class:
        raise ValueError(f"Unrecognized trainer: {trainer_type}")
    kwargs = load_kwargs(config)
    trainer_cls = trainer_nickname_to_trainer_class[trainer_type]
    return trainer_cls(config, **kwargs)
