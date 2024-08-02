import csv

conds =  ['sing', 'plu']
comparisons = ['grammatical', 'ungrammatical']
lemmas = {
    'BE': {'sing': 'is', 'plu': 'are'},
    'LIKE':{'sing': 'likes', 'plu': 'like'}
}

nouns = ['student', 'chameleon']

sentid = 0
contextid = 0
pairid = 0
stims = []
colnames = ['sentid', 'pairid', 'contextid', 'condition', 'comparison','sentence','ROI','expected']

expected = 'grammatical'

stims.append(colnames)

for i,cond in enumerate(conds): # condition
    for j, noun in enumerate(nouns): # context
        contextid +=1
        subj = nouns[j]
        obj = nouns[::-1][j]
        if cond == 'plu':
            subj +='s'
            obj +='s'
            roi = '1,2'
        else:
            subj = 'the '+ subj
            obj = 'the '+obj
            roi = '2,3'

        for lemma in lemmas: # pair
            pairid+=1
            verb_match = lemmas[lemma][conds[i]]
            verb_mismatch = lemmas[lemma][conds[::-1][i]]
            for comp in comparisons: #sent
                sentid+=1
                if comp == 'grammatical':
                    sent = f'{subj} {verb_match} {obj}.'
                else:
                    sent = f'{subj} {verb_mismatch} {obj}.'

                curr_stim = [sentid, pairid, contextid, cond, comp, sent, roi, expected]


                stims.append(curr_stim)



with open('../../stimuli/minimal_pairs.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    for row in stims:
        writer.writerow(row)


   



