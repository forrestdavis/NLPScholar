# Basic class for analyzing results from minimal pair (or TSE) experiments
# Implemented by Grusha Prasad
# (https://github.com/grushaprasad)
# August 2024

from .Analysis import Analysis
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' 

class MinimalPair(Analysis):

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)

        print('\n-----------------------')
        print("Processing config file\n")

       
        # Create condition column
        
        self.validcols = []
        for col in self.conditions.split(','): 
            col = col.strip()
            if col in self.conddat:
                self.validcols.append(col)
            elif len(col) > 0:
                print(f"WARNING: {col} not in conddat. Ignoring.")

        if len(self.validcols) == 0:
            print('WARNING: No valid condition columns entered')

        self.conddat['cond'] = self.conddat[self.validcols].apply(
            lambda row: '_'.join(row.astype(str)), axis=1
        )

        # Create output column 

        self.output = {'by_word': False,
                       'by_pair': False,
                       'by_cond': False}

        for out in self.save.split(','):
            if out.strip() in self.output:
                self.output[out.strip()] = True
            else:
                print(f'{out.strip()} is an invalid output type. Ignoring.')
        print('-----------------------')



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


    



    # def save_interim(self,target_wide_df):

    #     target_long = pd.melt(target_wide_df,
    #                           value_vars = ['expected', 'unexpected'], 
    #                           id_vars=['pairid','pos', 'model','condition','lemma'],
    #                           value_name = self.pred_measure)

        
    #     cols = ['model', 'pairid', 'lemma', 'condition', 'comparison']
    #     summ = target_long.groupby(cols).agg({self.pred_measure: ['sum','mean']}).reset_index()
    #     summ.columns = list(map(''.join, summ.columns.values))
    #     fname = f"{self.resultsfpath.replace('.tsv', '')}_byROI.tsv"
    #     print(f"Saving interim file: {fname}")
    #     summ.to_csv(fname, sep = '\t', na_rep='NA')




    def summarize_roi(self, by_word):

        if 'ROI' in by_word: ## want only specific regions
            by_word = by_word.astype({'ROI': 'string'}) # to avoid autocasting if ROIs have only one val

            # Exclude non ROI words
            by_word['ROI'] = by_word['ROI'].apply(lambda x: [int(item.strip()) for item in x.split(',')])

            by_word['target'] = by_word.apply(lambda x: True if x['wordpos_mod'] in x['ROI'] else False, axis=1)

            target = by_word.loc[by_word['target']] ## Creating subset for convenience. Will cause warnings that can be ignored.

            # Get consistent ROI mapping across comparisons
            ## Do I really want this if we just have equal? 
            # target['mapping'] = target.apply(lambda x: {curr:new for new, curr in enumerate(x['ROI'])}, axis=1)
            # target['pos'] = target.apply(lambda x: x['mapping'][x['wordpos_mod']], axis=1)

        else: # want entire sentence
            target = by_word
            target['pos'] = target['wordpos_mod']


        # Specify pred measure
        pred_measure = self.get_measure()

        # Summarize over all words in sentences

        groupby_cols = ['model', 'cond', 'pairid', 'comparison']

        target_summ = target.groupby(groupby_cols).agg({pred_measure: 'mean'}).reset_index()

        # Compute perplexity
        if self.pred_measure == 'perplexity':
            target_summ['surp'] = 2**target_summ['surp']

        # Make data wider

        by_pair = target_summ.pivot(index=['model', 'cond', 'pairid'], columns='comparison', values=pred_measure).reset_index()

        # Remove invalid pairs
        bad_pairs = set(by_pair[by_pair['expected'].isna() | by_pair['unexpected'].isna()]['pairid'])

        by_pair = by_pair[by_pair['expected'].notna() & by_pair['unexpected'].notna()]

        if len(bad_pairs) > 0: # removed some bad pairs
            print(f"WARNING: Excluding pairs which did not have expected or unexpected comparisons:\n{bad_pairs}\n")

        # Compute difference and accuracy
        by_pair['diff'] = by_pair['expected'] - by_pair['unexpected']

        by_pair['acc'] = self.get_acc(by_pair)

        return by_pair


    # def get_diff(self, dat, diff_type='macro'):
    #     if self.pred_measure == 'prob':
    #         return dat['expected'] - dat['unexpected']
    #     elif self.pred_measure == 'perplexity' and diff_type == 'micro':
    #         return  # cannot have perplexity of individual tokens
    #     else:
    #         return dat['unexpected'] - dat['expected']

    def get_acc(self, dat):
        if self.pred_measure == 'prob':
            return dat['expected'] > dat['unexpected']
        else:
            return dat['unexpected'] > dat['expected']

    def get_measure(self):
        if self.pred_measure == 'prob':
            return 'prob'
        else:
            return 'surp' # we need surp for perplexity too

    # def filter_lemma(self, target_wide):
    #     if self.pred_measure == 'prob':
    #         ascending = False
    #     else:
    #         ascending = True

    #     groupby_cols = ['contextid','model', 'condition']

    #     by_lemma = target_wide.sort_values(by=['expected'], ascending=ascending).groupby(groupby_cols).head(self.topk).reset_index(drop=True)

    #     return by_lemma



    # def summarize_lemma(self, by_pair):
    #     groupby_cols = ['contextid','model', 'condition']

    #     by_context = by_pair.groupby(groupby_cols).agg({'microdiff': 'mean', 'macrodiff': 'mean', 'acc': 'mean'}).reset_index()

    #     return by_context

    def summarize_cond(self, by_pair):
        groupby_cols = ['model', 'cond']

        by_cond = by_pair.groupby(groupby_cols).agg({'acc': 'mean', 'diff': 'mean',  'expected': 'mean', 'unexpected': 'mean'}).reset_index()

        by_cond['macrodiff'] = by_cond['expected'] - by_cond['unexpected']

        return by_cond


    def save_df(self, dat, fpath):
        # re-create condition columns
        if len(dat) == 0:
            print(f'Empty dataframe. Not creating {fpath}.')
        else:
            if len(self.validcols) > 0 and 'cond' in dat:
                dat[self.validcols] = dat['cond'].str.split('_', expand = True)

                # reorder columns 
                for colname in self.validcols[::-1]:
                    temp = dat.pop(colname)
                    dat.insert(1, colname, temp)

            # remove column created in pipeline
            dat = dat.drop(['cond'], axis=1)

            print(f'Creating {fpath}\n')

        dat.to_csv(fpath, sep = '\t', na_rep='NA')

    def analyze(self):
        """ Run Minimal Pair analysis and save result to resultsfpath

        The output has the following information. 
        - model: the model being evaluated
        - condition: different conditions
        - diff: average difference in predictability (default micro)
        - acc: proportion of times predictable > unpredictable (across all pairs of sentences in each condition)
        """

        # combine subword tokens to words (handling punctuation)
        by_word = self.token_to_word(self.preddat.copy())

        # combine with conddat (if present)
        if len(self.conddat) > 0:
            by_word = pd.merge(by_word,self.conddat, on='sentid')

        if self.output['by_word']:
            fname = self.resultsfpath.replace('.tsv', '_byword.tsv')
            self.save_df(by_word, fname)



        # comparison summarized over all ROIs for each pair 

        by_pair = self.summarize_roi(by_word)

        if self.output['by_pair']:
            fname = self.resultsfpath.replace('.tsv', '_bypair.tsv')
            self.save_df(by_pair, fname)


        # summarize over contexts

        if self.output['by_cond']:
            by_cond = self.summarize_cond(by_pair)
            self.save_df(by_cond, self.resultsfpath)

        
