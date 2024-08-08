import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.load_models import load_models

'''
accuracy = evaluate.load('precision')

config = {'models': 
          {'hf_text_classification_model':
           #["imdb_model"]},
           ["distilbert/distilbert-base-uncased"]},
          'loadPretrained': False,
          'num_labels': 5,
         }

classifier = load_models(config)[0]
text = ['he is so so happy', 'he is so so happy']
predictions =  classifier.get_text_output(text)['logits']
print(predictions)


sys.exit(1)
'''
id2label = {

    0: "O",

    1: "B-corporation",

    2: "I-corporation",

    3: "B-creative-work",

    4: "I-creative-work",

    5: "B-group",

    6: "I-group",

    7: "B-location",

    8: "I-location",

    9: "B-person",

    10: "I-person",

    11: "B-product",

    12: "I-product",

}

config = {'models': 
             {'hf_token_classification_model':
              ["ner_model"]},
          'device': 'mps', 
          'batchSize': 10, 
          "id2label": id2label}

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
import sys
sys.exit()

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
output = classifier.get_by_token_predictions(text)

for batch in output:
    for word in batch:
        print(classifier.tokenizer.convert_ids_to_tokens(word['token_id']), 
              word['label'],
              word['probability'])
    print('-'*80)


print()
config = {'models': 
          {'hf_text_classification_model':
           ["ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"]}}

text = "A black race car starts up in front of a crowd of people."
pair = "A man is driving down a lonely road."

classifier = load_models(config)[0]
print(classifier.get_text_predictions(text, pair))
