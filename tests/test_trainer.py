import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.trainers.TextClassification import TextClassificationTrainer
import torch

config = {'models': 
          {'hf_text_classification_model':
           ["distilbert/distilbert-base-uncased"]}, 
         }

kwargs = {'device': 'mps', 
          'loadPretrained': False, 
          'num_labels': 2,
          'batchSize': 10, 
          'trainfpath': 'imdb:train', 
          'validfpath': 'imdb:test'}

trainer = TextClassificationTrainer(config, 
                                    **kwargs)

trainer.train()
#dataset = trainer.load_from_hf_dataset('imdb', 'train').to_pandas()
#dataset = trainer.load_from_tsv('imdb_train.tsv')

#print(dataset)

#dataset.to_csv('imdb_train.tsv', sep='\t', index=False)


