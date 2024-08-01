from src.models.load_models import load_models
from src.experiments.TSE import TSE

config = {'models': 
             {'hf_masked_model': ['bert-base-cased']}, 
          'tokenizers':
            {'hf': ['bert-base-cased']}, 
          'ignorePunct': None, 
          'predfpath': 'results/minimal_pairs.tsv',
          'condfpath': 'stimuli/minimal_pairs.tsv',
          'device': 'mps'}

experiment = TSE(config)
experiment.run()

