import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.evaluations.MinimalPair import MinimalPair
from src.evaluations.TextClassification import TextClassification
from src.evaluations.TokenClassification import TokenClassification

config = {'models': 
             {'hf_masked_model': ['bert-base-cased']}, 
          'ignorePunct': None, 
          'predfpath': 'results/minimal_pairs.tsv',
          'datafpath': 'data/minimal_pairs.tsv'}

config = {'models': 
          {'hf_text_classification_model':
           ["ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"]}}

kwargs = {
          'loadAll': True,
          'device': 'mps', 
          'batchSize': 10}

config = {'models': 
             {'hf_token_classification_model':
              ["QCRI/bert-base-multilingual-cased-pos-english"]},
          'predfpath': 'results/pos.tsv',
          'datafpath': 'data/pos.tsv',
          'device': 'mps', 
          'batchSize': 10}

#evaluation = TextClassification(config, **kwargs)
#evaluation = MinimalPair(config, **kwargs)
#evaluation = Interact(config)
#evaluation = TextClassification(config)
config = {**config, **kwargs}
evaluation = TokenClassification(config)

evaluation.interact()

