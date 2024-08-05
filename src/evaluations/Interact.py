# Basic class for interactive interface for LMs
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024

from ..utils.load_models import load_models

class Interact:

    def __init__(self, config: dict):

        self.LM = load_models(config)[0]

    def run(self):

        word = "word" + ' '*16
        split = "Split" 
        unk = "Unk"
        model = "ModelName" + ' '*11
        surp = "surp" + ' '*4
        prob = "prob" + ' '*6
        header = f"{word} | {split} | {unk} | {model} | {surp} | {prob}"

        while True:
            sent = input('string: ').strip()
            print(header)
            print('-'*len(header))
            output = self.LM.get_aligned_words_predictabilities(sent)[0]
            for word in output:
                surp = round(word.surp, 3)
                prob = round(word.prob, 5)
                print_out = f"{word.word: <20} | {word.isSplit:5} | {word.isUnk:3} | {word.modelName.split('/')[-1]: <20} | {surp: >8} | {prob: >10}"
                print(print_out)        
