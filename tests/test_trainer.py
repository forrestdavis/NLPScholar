import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.load_trainers import load_trainer
import torch

config = {'exp': 'TextClassification', 
          'mode': ['train'], 
          'models': 
              {'hf_text_classification_model': ['distilbert-base-uncased']}, 
          'trainfpath': 'imdb:train',
          'validfpath': 'imdb:test',
          'modelfpath': 'imdb_model',
          'loadPretrained': False, 
          'textLabel': 'premise', 
          'pairLabel': 'hypothesis', 
          'numLabels': 3,
          'device': 'cpu',
          'samplePercent': 10}

trainer = load_trainer(config)
trainer.set_dataset()
print(trainer.dataset)
#trainer.preprocess_dataset()
#trainer.show_k_samples(3)
#print('ok')
#trainer.train()
#dataset = trainer.load_from_hf_dataset('imdb', 'train').to_pandas()
#dataset.to_csv('imdb_train.tsv', sep='\t', index=False)

#dataset = trainer.load_from_tsv('imdb_train.tsv')

#print(dataset)

#dataset.to_csv('imdb_train.tsv', sep='\t', index=False)


