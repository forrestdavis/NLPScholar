import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.classifiers.hf_text_classification_model import HFTextClassificationModel
from src.classifiers.load_classifiers import load_classifiers

config = {'models': 
             {'hf_text_classification_model': ['stevhliu/my_awesome_model']}, 
          'predfpath': 'results/minimal_pairs.tsv',
          'condfpath': 'stimuli/minimal_pairs.tsv',
          'device': 'mps', 
          'batchSize': 10}

classifier = HFTextClassificationModel("stevhliu/my_awesome_model", None)
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text = [text, 'I loved it']

print(classifier.get_predictions(text))

print()

classifier = load_classifiers(config)[0]
print(classifier.get_predictions(text))

