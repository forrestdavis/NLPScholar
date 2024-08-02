import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.experiments.TSE import TSE
from src.experiments.Interact import Interact
from src.experiments.TextClassification import TextClassification

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
             {'hf_text_classification_model':
              ['ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli']},
          'predfpath': 'results/snli_paired_sentences.tsv',
          'condfpath': 'stimuli/snli_paired_sentences.tsv',
          'device': 'mps', 
          'batchSize': 10}

#experiment = TSE(config)
#experiment = Interact(config)
experiment = TextClassification(config)

experiment.run()

