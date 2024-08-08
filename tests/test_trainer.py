import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.trainers.HFTextClassificationTrainer import HFTextClassificationTrainer
import torch

config = {'models': 
          {'hf_text_classification_model':
           ["distilbert/distilbert-base-uncased"]}, 
         }

kwargs = {'device': 'mps', 
          'loadPretrained': False, 
          'num_labels': 3,
          'batchSize': 10, 
          'textLabel': 'premise', 
          'pairLabel': 'hypothesis',
          #'trainfpath': 'imdb:train', 
          #'validfpath': 'imdb:test',
          #'modelfpath': 'imdb_model',
          'trainfpath': 'nyu-mll/glue:mnli:train', 
          'validfpath': 'nyu-mll/glue:mnli:validation_matched', 
          'modelfpath': 'mnli_model',
          'device': 'mps',
         }

trainer = HFTextClassificationTrainer(config, 
                                    **kwargs)

trainer.set_dataset()
trainer.preprocess_dataset()
trainer.show_k_samples(3)
#print('ok')
#trainer.train()
#dataset = trainer.load_from_hf_dataset('imdb', 'train').to_pandas()
#dataset.to_csv('imdb_train.tsv', sep='\t', index=False)

#dataset = trainer.load_from_tsv('imdb_train.tsv')

#print(dataset)

#dataset.to_csv('imdb_train.tsv', sep='\t', index=False)


