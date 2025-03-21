# Basic class for getting word level surprisal
# Implemented by Grusha Prasad
# (https://github.com/grushaprasad)
# March 2025

from .Analysis import Analysis
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' 

class WordPredictability(Analysis):

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)

    def token_to_word(self, preddat):

        # Change word positions of punctuation
        self.handle_punctuation(preddat)

        # Summarize over word position
        groupby_cols = ['sentid', 'wordpos_mod', 'model']

        summ = preddat.groupby(groupby_cols).agg({'prob': 'mean', 'surp': self.word_summary}).reset_index() #note prob is always mean, cannot be sum. 

        # Realign wordpositions (remove gaps from handling punctuation)

        summ = summ.groupby(['model']).apply(self.remove_gaps, colname = 'wordpos_mod', include_groups=False).reset_index()

        return summ

    def handle_punctuation(self, dat):
        dat['wordpos_mod'] = dat['wordpos']
        if self.punctuation == 'previous':
            dat.loc[dat['punctuation']==True, 'wordpos_mod'] = dat.loc[dat['punctuation']==True, 'wordpos']-1
        elif self.punctuation == 'next':
            dat.loc[dat['punctuation']==True, 'wordpos_mod'] =  dat.loc[dat['punctuation']==True, 'wordpos']+1

    def remove_gaps(self, grouped_df, colname):

        prev=-1 #manually keep track because row indices not consecutive in grouped dataframe

        for i, row in grouped_df.iterrows():
            if prev != -1: #skip first row in grouped_df
                diff = row[colname]-grouped_df.loc[prev, colname]
                if diff>1:
                    grouped_df.loc[i,colname]-=(diff-1)

            prev=i
        return grouped_df

    def analyze(self):
        """ Get word level predictability
        """

        # combine subword tokens to words (handling punctuation)
        by_word = self.token_to_word(self.preddat.copy())
        by_word = by_word.rename(columns = {'wordpos_mod': 'word_pos'})

        print(f'Creating {self.resultsfpath}\n')


        ## To do: 

        # Merge with condition so minimially the sentence is there
        # Maybe try to get the individual words as well ? 

        merged = pd.merge(self.conddat, by_word, on=['sentid', 'word_pos'])
 
        merged = merged.drop("level_1", axis=1)

        merged.to_csv(self.resultsfpath, sep = '\t', na_rep='NA') 







