import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.experiments.TSE import TSE
from src.experiments.Interact import Interact
from src.experiments.TextClassification import TextClassification
from src.experiments.TokenClassification import TokenClassification

config = {'models': 
             {'hf_masked_model': ['bert-base-cased']}, 
          'tokenizers':
            {'hf': ['bert-base-cased']}, 
          'ignorePunct': None, 
          'predfpath': 'results/minimal_pairs.tsv',
          'condfpath': 'stimuli/minimal_pairs.tsv',
          'device': 'mps', 
          'batchSize': 10}

config = {'models': 
             {'hf_token_classification_model':
              ["QCRI/bert-base-multilingual-cased-pos-english"]},
          'predfpath': 'results/pos.tsv',
          'condfpath': 'stimuli/pos.tsv',
          'device': 'mps', 
          'batchSize': 10}

#experiment = TSE(config)
#experiment = Interact(config)
#experiment = TextClassification(config)
experiment = TokenClassification(config)

experiment.run()

