# Basic parent class for analyzing results
# Implemented by Grusha Prasad
# (https://github.com/grushaprasad)
# September 2024

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' 

class Analysis:
    def __init__(self, config:dict, **kwargs):

        # Default values
        self.predfpath = False
        self.datafpath = False
        self.resultsfpath = False
        self.sep = '\t' #separator

        #Params relevant only for MinimalPair
        self.pred_measure = 'surp'
        self.word_summary = 'mean'
        # self.roi_summary = 'micro' # bring this back later maybe
        self.k_lemmas = 'all'
        self.punctuation = 'previous'
        self.save_byword = False
        self.save_bypair = False
        self.save_bycond = False
        self.conditions = ''
        self.save = ''
         
        # Set params
        for k, v in kwargs.items():
            setattr(self, k, v)


        # Load data
        self.preddat = pd.read_csv(self.predfpath, sep=self.sep)

        if self.datafpath: 
            self.conddat = pd.read_csv(self.datafpath, sep=self.sep)
        else: # not required if only want byword
            print('No condition path specified')
            self.conddat = pd.DataFrame() 

        


