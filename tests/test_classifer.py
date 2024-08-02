import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.classifiers.hf_text_classification_model import HFTextClassificationModel
from src.classifiers.load_classifiers import load_classifiers

config = {'models': 
             {'hf_text_classification_model':
              ['ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli']},
          'predfpath': 'results/minimal_pairs.tsv',
          'condfpath': 'stimuli/minimal_pairs.tsv',
          'device': 'mps', 
          'batchSize': 10}

classifier = load_classifiers(config)[0]
premise = ["A man inspects the uniform of a figure in some East Asian country.", 
           "An older and younger man smiling.", 
           "A black race car starts up in front of a crowd of people.", 
           "A soccer game with multiple males playing.", 
           "A smiling costumed woman is holding an umbrella."]

hypothesis = ["The man is sleeping.",
              "Two men are smiling and laughing at the cats playing on the "\
              "floor.", 
              "A man is driving down a lonely road.", 
              "Some men are playing a sport.",
              "A happy woman in a fairy costume holds an umbrella."]

print(classifier.get_text_predictions(premise, hypothesis))

print()

'''
classifier = load_classifiers(config)[0]
print(classifier.get_predictions(text))

'''
