# Basic class for analyzing results from minimal pair (or TSE) experiments
# Implemented by Grusha Prasad
# (https://github.com/grushaprasad)
# August 2024

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' 

class MinimalPairAnalysis:
    def __init__(self, config: dict):


        # Check for necessary parameters 

        for arg in ['predfpath', 'datafpath', 'resultsfpath', 'pred_measure']:
            assert arg in config, f"Must pass in {arg}"
            val = config[arg]
            exec(f"self.{arg} = '{val}'")
        
        # Assign optional parameters
        defaults = {
            'word_summary': 'mean',
            'roi_summary': 'micro',
            'k_lemmas': 'inf', ## include all lemmas
            'punctuation': 'previous'
        }

        for arg in defaults:
            if arg in config:
                val = config[arg]
            else:
                val = defaults[arg]
            exec(f"self.{arg} = '{val}'")


        # Load data
        self.preddat = pd.read_csv(self.predfpath, sep='\t')

        self.conddat = pd.read_csv(self.datafpath, sep='\t')

        # Set topk

        try:
            self.topk = int(self.k_lemmas.strip())
        except:
            if self.k_lemmas != 'all':
                print('\nINVALID TOP-K LEMMAS, USING ALL LEMMAS IN CALCULATION\n')

            self.topk = len(self.conddat['lemma'].unique())



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


    def summarize_roi(self, by_word):
        merged = pd.merge(by_word,self.conddat, on='sentid')

        # Exclude non ROI words
        merged['ROI'] = merged['ROI'].apply(lambda x: [int(item.strip()) for item in x.split(',')])

        merged['target'] = merged.apply(lambda x: True if x['wordpos_mod'] in x['ROI'] else False, axis=1)

        target = merged.loc[merged['target']] ## Creating subset for convenience. Will cause warnings that can be ignored.

        # Get consistent ROI mapping across comparisons
        target['mapping'] = target.apply(lambda x: {curr:new for new, curr in enumerate(x['ROI'])}, axis=1)
        # target['wordpos_mod'] = target['wordpos_mod'].apply(lambda x: str(x))
        target['pos'] = target.apply(lambda x: x['mapping'][x['wordpos_mod']], axis=1)

        # Specify pred measure
        pred_measure = self.get_measure()

        # Compare pairs for each ROI (micro measures, accuracy)
        groupby_cols = ['pairid', 'model', 'condition', 'lemma', 'contextid']

        target_wide = target.pivot(index=['pairid', 'pos', 'model', 'condition','lemma', 'contextid'], columns='comparison', values=pred_measure).reset_index()

        target_wide['microdiff'] = self.get_diff(target_wide, 'micro')
        target_wide['acc'] = self.get_acc(target_wide)

        # Summarize over ROIs
        by_pair = target_wide.groupby(groupby_cols).agg({'expected': 'mean', 'unexpected': 'mean', 'microdiff': 'mean', 'acc': 'mean'}).reset_index()

        # Compute perplexity
        if self.pred_measure == 'perplexity':
            by_pair['expected'] = 2**by_pair['expected']
            by_pair['unexpected'] = 2**by_pair['unexpected']

        # Compute macro measures
        by_pair['macrodiff'] = self.get_diff(by_pair, 'macro')


        return by_pair


    def get_diff(self, dat, diff_type='macro'):
        if self.pred_measure == 'prob':
            return dat['expected'] - dat['unexpected']
        elif self.pred_measure == 'perplexity' and diff_type == 'micro':
            return  # cannot have perplexity of individual tokens
        else:
            return dat['unexpected'] - dat['expected']

    def get_acc(self, dat):
        if self.pred_measure == 'prob':
            return dat['expected'] > dat['unexpected']
        else:
            return dat['unexpected'] > dat['expected']

    def get_measure(self):
        if self.pred_measure == 'prob':
            return 'prob'
        # elif self.pred_measure == 'perplexity':
        #     return('surp', 'sum') #we need sum surprisal (or is it mean?)
        else:
            return 'surp'

    def summarize_lemma(self, by_pair):

        if self.pred_measure == 'prob':
            ascending = False
        else:
            ascending = True

        groupby_cols = ['contextid','model', 'condition']

        print('PRINTING TOP K', self.topk)

        by_lemma = by_pair.sort_values(by=['expected'], ascending=ascending).groupby(groupby_cols).head(self.topk).reset_index(drop=True)


        by_context = by_lemma.groupby(groupby_cols).agg({'microdiff': 'mean', 'macrodiff': 'mean', 'acc': 'mean'}).reset_index()

        return by_context

    def summarize_context(self, by_context):
        groupby_cols = ['model', 'condition']

        by_cond = by_context.groupby(groupby_cols).agg({'microdiff': 'mean', 'macrodiff': 'mean', 'acc': 'mean'}).reset_index()

        return by_cond


    def run(self):
        """ Run Minimal Pair analysis and save result to resultsfpath

        The output has the following information. 
        - model: the model being evaluated
        - condition: different conditions
        - diff: average difference in predictability (default micro)
        - acc: proportion of times predictable > unpredictable (across all pairs of sentences in each condition)
        """

        # combine subword tokens to words (handling punctuation)
        by_word = self.token_to_word(self.preddat.copy())

        # comparison summarized over all ROIs for each pair 
        by_pair = self.summarize_roi(by_word)

        # summarize over lemmas
        by_context = self.summarize_lemma(by_pair)

        # summarize over contexts
        by_cond = self.summarize_context(by_context)

        by_cond.to_csv(self.resultsfpath, sep = '\t', na_rep='NA')


## Tests

config = {
    'predfpath': '../../../predictions/minimal_pairs.tsv',
    'datafpath': '../../../data/minimal_pairs.tsv',
    'resultsfpath': '../../../results/minimal_pairs.tsv',
    'pred_measure': 'surp',
    'roi_summary': 'micro',
    'k_lemmas': '1'
}

x = MinimalPairAnalysis(config)

# from pprint import pprint
# pprint(vars(x))

x.run()

