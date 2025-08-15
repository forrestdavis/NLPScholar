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
        self.punctuation = 'previous'
        self.conditions = ''
        self.save = ''

        #Params relevant only for Classification
        self.f_val = 1
         
        # Set params
        for k, v in kwargs.items():
            setattr(self, k, v)


        # Load data
        print('\n-----------------------')
        print("Loading data")
        self.preddat = pd.read_csv(self.predfpath, sep=self.sep)

        if self.datafpath: 
            self.conddat = pd.read_csv(self.datafpath, sep=self.sep)
        else: # not required if only want byword
            print('No condition path specified')
            self.conddat = pd.DataFrame() 
        print('-----------------------')

        # Create condition column
        print('\n-----------------------')
        print("Verifying conditions\n")
        self.validcols = []
        for col in self.conditions.split(','): 
            col = col.strip()
            if col in self.conddat:
                self.validcols.append(col)
            elif len(col) > 0:
                print(f"WARNING: {col} not in data file. Ignoring.")

        if len(self.validcols) == 0:
            print('WARNING: No valid condition columns entered')

        self.conddat['cond'] = self.conddat[self.validcols].apply(
            lambda row: '_'.join(row.astype(str)), axis=1
        )
        print('-----------------------')

        


