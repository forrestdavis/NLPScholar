import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.load_models import load_models

config = {'models': 
             {'hf_token_classification_model':
              ["stevhliu/my_awesome_wnut_model"]},
          'device': 'mps', 
          'batchSize': 10}

classifier = load_models(config)[0]
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

text = ["The Golden State Warriors are an American professional basketball "\
        "team based in San Francisco.", "Kleinfeltersville is a town in "\
        "Pennsylvania."]

output = classifier.get_by_token_predictions(text)

for batch in output:
    for word in batch:
        print(classifier.tokenizer.convert_ids_to_tokens(word['token_id']), 
              word['label'],
              word['probability'])
    print('-'*80)


print()

'''
classifier = load_classifiers(config)[0]
print(classifier.get_predictions(text))

'''

config = {'models': 
             {'hf_token_classification_model':
              ["QCRI/bert-base-multilingual-cased-pos-english"]},
          'device': 'mps', 
          'batchSize': 10}

text = ["The Golden State Warriors are an American professional basketball "\
        "team based in San Francisco.", "Noam Chomsky is a soft revolution."]

classifier = load_models(config)[0]
output = classifier.get_token_predictions(text)

for batch in output:
    for word in batch:
        print(classifier.tokenizer.convert_ids_to_tokens(word['token_id']), 
              word['label'],
              word['probability'])
    print('-'*80)

