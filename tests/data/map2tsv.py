import pandas as pd 

fname = 'simple_agrmt.txt'

data = pd.read_csv(fname, sep='\t', header=None)
truths = data[0].tolist()
sentences = data[1].tolist()

output = {'sentid': [],
          'pairid': [], 
          'contextid': [], 
          'lemma': [], 
          'condition': [],
          'comparison': [],
          'sentence': [], 
          'ROI': []}

sentid = 0
pairid = 0
contextid = 0
condition = fname.split('.txt')[0]

for i in range(0, len(truths), 2):
    assert truths[i] and not truths[i+1]
    good = sentences[i]
    bad = sentences[i+1]
    good_words = good.split(' ')
    bad_words = bad.split(' ')

    ROI = 0
    for idx, (good_word, bad_word) in enumerate(zip(good_words, bad_words)):
        if good_word != bad_word:
            ROI = idx
            lemma = good_word
            break

    # Add good
    output['sentid'].append(sentid)
    output['pairid'].append(pairid)
    output['contextid'].append(contextid)
    output['lemma'].append(lemma)
    output['condition'].append(condition)
    output['comparison'].append('expected')
    output['sentence'].append(good)
    output['ROI'].append(ROI)

    # Add bad
    sentid += 1
    output['sentid'].append(sentid)
    output['pairid'].append(pairid)
    output['contextid'].append(contextid)
    output['lemma'].append(lemma)
    output['condition'].append(condition)
    output['comparison'].append('unexpected')
    output['sentence'].append(bad)
    output['ROI'].append(ROI)


    contextid += 1
    pairid += 1


output = pd.DataFrame.from_dict(output)
output.to_csv(fname.replace('.txt', '.tsv'), index=False, sep='\t')
