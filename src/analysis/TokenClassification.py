# Basic class for analyzing results from Token Classification experiments
# Implemented by Grusha Prasad
# (https://github.com/grushaprasad)
# September 2024

from .Analysis import Analysis
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' 

class TokenClassification(Analysis):
    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)

        # Create output column 
        print("Verifying output types\n")
        self.output = {'by_word': False,
                       'by_tokentype': False,
                       'by_cond': False}

        for out in self.save.split(','):
            if out.strip() in self.output:
                self.output[out.strip()] = True
            else:
                print(f'{out.strip()} is an invalid output type. Ignoring.')
        print('-----------------------')

        ## combine with conddat if possible
        if len(self.conddat) > 0:
            if 'target' in self.conddat:
                self.conddat = self.conddat.drop('target', axis=1) #because target is in both preddat and conddat
            self.dat = pd.merge(self.preddat, self.conddat, on='textid')
        else:
            self.dat = self.preddat

        # Create ignore tokens
        ignore_tokens = self.ignore.split(',')
        self.ignore = [x.strip() for x in ignore_tokens]



    def get_word_pred(self, df):
        """
        For given word in a sentence with possibly many sub-word tokens, returns df for the word based on self.agg_type

        """
        if self.agg_type == 'first':
            return df.head(1)
        elif self.agg_type == 'max':
            max_prob = df['prob'].max()
            return df.loc[df['prob'] == max_prob]
        else:
            print("WARNING: Invalid agg_type. Returning full data frame")
            return df

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

            # remove columns created in pipeline
            drop_cols = list(dat.columns[dat.columns.str.startswith('level_')])
            drop_cols += ['cond']

            dat = dat.drop(drop_cols, axis=1)

            print(f'Creating {fpath}\n')

            dat.to_csv(fpath, sep = '\t', na_rep='NA')

    def filter(self, dat):
        # optionally remove punctuation

        # remove any targets to ignore
        filtered = dat[~dat['target'].isin(self.ignore)]

        if self.ignore_punctuation:
            print('removing punctuation')
            print(len(filtered))
            filtered = filtered[filtered['punctuation']==False]
            print(len(filtered))



        return filtered

    def compute_measures(self, dat, average_types):
        from sklearn.metrics import precision_score, recall_score, fbeta_score, accuracy_score

        classes = dat['target'].unique()

        summ = pd.DataFrame({'target_class':classes})

        for avg in average_types:
        
            precision = precision_score(dat['target'], dat['predicted'], average=avg, labels = classes,  zero_division=0)
            recall = recall_score(dat['target'], dat['predicted'], average=avg, labels = classes,  zero_division=0)
            fscore = fbeta_score(dat['target'], dat['predicted'], beta=self.f_val, average=avg, labels=classes,  zero_division=0)
            accuracy = accuracy_score(dat['target'], dat['predicted'])

            if avg:
                avg_name = f'{avg}-'
            else:
                avg_name = ''

            summ = summ.assign(**{
                'target_class': classes,
                f'{avg_name}precision': precision,
                f'{avg_name}recall': recall,
                f'{avg_name}f{self.f_val}': fscore}
            )

        if None not in average_types:
            summ = summ.drop('target_class', axis=1)
            summ = summ.drop_duplicates()
            summ['accuracy'] = accuracy


        return summ

    def analyze(self):
        ## aggregate over subword tokens
        by_word = self.dat.groupby(['textid', 'wordpos', 'model']).apply(self.get_word_pred, include_groups=False).reset_index()

        if self.output['by_word']:
            fname = self.resultsfpath.replace('.tsv', '_byword.tsv')
            self.save_df(by_word, fname)

        ## Filter punctuation and unrelated before aggregation

        filtered = self.filter(by_word)

        if self.output['by_tokentype']:
            by_tokentype = filtered.groupby(['model','cond']).apply(self.compute_measures, [None], include_groups=False).reset_index()

            fname = self.resultsfpath.replace('.tsv', '_bytokentype.tsv')
            self.save_df(by_tokentype, fname)

        if self.output['by_cond']:
            by_cond = filtered.groupby(['model','cond']).apply(self.compute_measures, ['micro','macro'], include_groups=False).reset_index()

            fname = self.resultsfpath.replace('.tsv', '_bycond.tsv')
            self.save_df(by_cond,fname)





















