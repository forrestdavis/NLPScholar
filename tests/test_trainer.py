import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.trainers.HFTokenClassificationTrainer import HFTokenClassificationTrainer
from src.trainers.HFTextClassificationTrainer import HFTextClassificationTrainer
import torch

config = {'models': 
          {'hf_token_classification_model':
           ["distilbert/distilbert-base-uncased"]}, 
         }

kwargs = {'device': 'mps', 
          'loadPretrained': False, 
          'num_labels': 13,
          'batchSize': 64, 
          'epochs': 10,
          #'textLabel': 'premise', 
          #'pairLabel': 'hypothesis',
          #'trainfpath': 'imdb:train', 
          #'validfpath': 'imdb:test',
          #'modelfpath': 'imdb_model',
          'trainfpath': 'wnut_17:train',
          'validfpath': 'wnut_17:test',
          'modelfpath': 'ner_model',
          'device': 'mps',
         }

trainer = HFTokenClassificationTrainer(config, 
                                    **kwargs)

#trainer.set_dataset()
#trainer.preprocess_dataset()
#trainer.show_k_samples(3)
#print('ok')
trainer.train()
#dataset = trainer.load_from_hf_dataset('imdb', 'train').to_pandas()
#dataset.to_csv('imdb_train.tsv', sep='\t', index=False)

#dataset = trainer.load_from_tsv('imdb_train.tsv')

#print(dataset)

#dataset.to_csv('imdb_train.tsv', sep='\t', index=False)


