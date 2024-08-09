import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.trainers.HFTokenClassificationTrainer import HFTokenClassificationTrainer
from src.trainers.HFTextClassificationTrainer import HFTextClassificationTrainer
from src.trainers.HFLanguageModelTrainer import HFLanguageModelTrainer
import torch

config = {'models': 
          #{'hf_causal_model': ['gpt2']},
          {'hf_masked_model':['distilbert-base-uncased']},
         }

kwargs = {'device': 'mps', 
          'loadPretrained': False, 
          #'numLabels': 13,
          'batchSize': 64, 
          'epochs': 3,
          'precision': '16bit',
          #'textLabel': 'premise', 
          #'pairLabel': 'hypothesis',
          #'trainfpath': 'imdb:train', 
          #'validfpath': 'imdb:test',
          #'modelfpath': 'imdb_model',
          #'trainfpath': 'wnut_17:train',
          #'validfpath': 'wnut_17:test',
          #'modelfpath': 'ner_model',
          'trainfpath': 'wikitext:wikitext-2-raw-v1:train',
          'validfpath': 'wikitext:wikitext-2-raw-v1:validation',
          'modelfpath': 'wiki2model',
          'device': 'mps',
         }

trainer = HFLanguageModelTrainer(config, 
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


