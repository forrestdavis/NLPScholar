# MinimalPair Analysis

The `analysis` mode of `MinimalPair` can be used in two ways: 

1. Get by-word predictability estimates from LMs from a dataset. 

2. Compare the predictability of words in minimally different contexts. 

For 2, we assume that all comparisons are binary (i.e., in any given comparison we are comparing two words or the same word in two contexts). Some examples of binary comparisons are grammatical vs. ungrammatical, ambiguous vs. unambiguous or animate vs. inanimate. 

A minimal-pair analysis requires the experimeter to specify an expected direction, e.g., "predictability of grammatical is greater than ungrammatical". 


In this toolkit:

- "predictability" can either refer to conditional probability --- $P(word | context)$ --- or surprisal --- $-\log P(word | context)$

- In measuring predictability of a "word" we either average or sum across the predictability of all the sub-word tokens depending on the specified parameters (see below). 

- "context" can either be just the left context (autoregressive LMs) or both left and right context (Masked LMs). 

- What counts as "minimally different" is specified by the experimenter. 


# Input requirements

## TSV file with predictability (file passed into predfpath)
This is the ouptut of the `evaluate` mode. It should have one sub-word token per line for every pair of sentences that needs to be compared for every model. The tsv has the following columns. 

- `token` (the subword token)
- `sentid` (the ID of the sentences)
- `word` (the word the subword token belongs to)
- `wordpos` (the specific word in the context sentence that the subword token belongs to)
- `model` (the name of the language model)
- `tokenizer` (the name of the tokenizer)
- `punctuation` (is the token punctuation?)
- `prob` (the probability of the token given the context)
- `surp` (the surprisal of the token given the context; log base 2)

## TSV file with conditions (file passed into datafpath) 

**This file is not required if you want to only get by-word predictability estimates.** 

- `sentid` (the ID of the specific sentence; matches what is in `predictability.tsv`)
- `comparison` (expected and unexpected)
- `pairid` (the ID of the specific pair of sentences being compared)
- `sentence` (the full sentence)
- `ROI` (the specific word positions that are of interest; optional)
- Any other specific conditions or groups that the sentence belongs to. 

**Note on ROI**: In many cases, ROIs are just single words, in which case this field is filled with just one integer corresponding to the target word position. Some experiments might have multiple words of interest. In such cases, each of the words should be comma separated. Note: it is ok for the ROIs to differ across pairs. 

## Parameters to be included in the config:

### Required parameters

- `predfpath`: File path and name for predictability TSV
- `datafpath`: File path and name for data TSV
- `resultsfpath`: File path and name for output TSV
- `save`: The final files that should be saved. This can be `by_word`, `by_pair`, and/or `by_cond`. If you want multiple files, they should be comma separated. 

### Optional parameters
- `pred_measure`: `prob`, `surp`, `perplexity`. Specfies which measure to use in the output results; perplexity is sentence level. (Default is `surp`)
- `word_summary`: `sum` or `mean` predictability across all sub-word tokens. (Default is mean; Probability always uses `sum`)
- `punctuation`: `previous`, `next`, `separate`, `ignore` (how to handle punctation tokens; fold into previous word, next word, treat as a separate word or ignore from computation. If it is treated as separate token, this will influence ROIs.)
- `conditions`: the names of the columns in `datafpath` that should be treated as conditions for `by_cond`. Multiple column names be comma separated.  If no conditions are specified, the output of `by_cond` will be aggregated over all rows.   


# Output

The output is 
### `by_word`
A TSV file with one row per word in the dataset per model. Predictability for each word is either averaged or summed across the sub-word tokens (as specified by the `word_summary` flag). Punctuation is either ignored or folded into the previous/next word (as specified by the `puntuation` tag).  

### `by_pair`
 A TSV file with one row per pair of sentences in the dataset per model, with separate columns for `expected` and `unexpected`. The predictability estimates for these columns is computed by averaging over all the words in the `ROI` (or the entire sentence if `ROI` is not specified). This also has columns for difference and accuracy indicating whether `expected` is more predictable than `unexpected`.

### `by_cond`

2. A TSV file with one row for each condition and model combination. Values in the `expected` and `unexpected` columns are the average over all pairs. Values in the `diff` column is the micro-average difference over all pairs (i.e., the average of differences). Values in the `acc` column are the proportion of pairs where `expected` was more predictable than `unexected. `
