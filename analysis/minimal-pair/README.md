# Defining Minimal-pair analysis [SHOULD THIS BE TSE ANALYSIS ??]

We define a minimal pair analysis as any analysis that either: 

1. Compares the predictability of minimally different words in the same context.

2. Compares the predictability of the same word in minimally different contexts. 

We assume that all comparisons are binary (i.e., in any given comparison we are comparing two words or the same word in two contexts). Some examples of binary comparisons are grammatical vs. ungrammatical, ambiguous vs. unambiguous or animate vs. inanimate. 

A minimal-pair analysis requires the experimeter to specify an expected direction, e.g., "predictability of grammatical is greater than ungrammatical". 


In this toolkit:

- "predictability" can either refer to conditional probability --- $P(word | context)$ --- or surprisal --- $-\log P(word | context)$

- In measuring predictability of a "word" we either average or sum across the predictability of all the sub-word tokens depending on what is specified in `params.py` (see below). 

- "context" can either be just the left context (autoregressive LMs) or both left and right context (Masked LMs). 

- What counts as "minimally different" is specified by the experimenter. 



# Input requirements

## TSV file with predictability
This has one sub-word token per line for every pair of sentences that needs to be compared. The tsv has the following columns. 
- `token` (the subword token)
- `sentid` (the ID of the specific pair of sentences being compared)
- `wordpos` (the specific word in the context sentence that the subword token belongs to)
- `comparison` (the specific comparison condition, e.g., "grammatical" or "ungrammatical")
- `prob` (the probability of the token given the context)
- `surp` (the surprisal of the token given the context; log base 2)
- `punctuation` (is the token punctuation?)

Note, this setup assumes that if an experiment wants to compare multiple word-pairs in the same context each of the word-pairs gets a different sentence ID. For example, consider the following context from Newman et al: 

```The keys to the cabinet ____ on the table. ```

Given the context, if one wanted to compare the probability of `___` being filled with is vs. are, but also exist vs. exists, the resulting tsv file would have two sentids: one for each of the lemmas BE and EXIST. Each sentid would have two unqiue sentences: one for each of the comparisons grammatical and ungrammatical. If there were 1000 lemmas, then there would be 1000 sentids, and 2000 sentences in the tsv. 


## TSV file with conditions

- `sentid` (the ID of the specific pair of sentences being compared; matches what is in `predictability.tsv`)
- `sentence` (the full sentence)
- `lemma` (the word lemma)
- `contextid` (the ID of context shared across all lemmas)
- `condition` (the specific conditions or group that the sentence belongs to)
- `ROI` (the specific word positions that are of interest)
- `expected` (the comparison that is predicted to have higher probability; comparison should match what is in `predictability.tsv`)

All sentences in a `condition` will be averaged together in the final output. There will be as many rows in the final output as there are conditions. For example, each of the "templates" in Marvin and Linzen (2018) and all of the "pheonomenon" in Warstadt et al (2020) can be thought of as conditions. 

In many cases, ROIs are just single words, in which case this field is filled with just one integer corresponding to the target word position. If the ROIs are different for each of the comparisons, they should be separated by a semi-colon, with the expected comparison coming first. For example, if are two comparisons, ambiguous and unambiguous and: 
- Unambiguous is expected to have higher predictability
- The target words in ambiguous occurs at positions 6 and the target words in unambiguous occur at position 9.

The ROI filed in this case should be filled with  `6;9`

Some experiments might have multiple words of interest. In such cases, each of the words should be comma separated. For example, if the target words in ambiguous occur at positions 6--8 and the target words in unambiguous occur at positions 9--11, the ROI field should be filled with: `9,10,11;6,7,8`. 


## Other parameters inputted with flags:

- `predfpath`: File path and name for predictability TSV
- `condfpath`: File path and name for conditons TSV
- `outfpath`: File path and name for output TSV
- `predicability-measure`: `probability`, `surprisal`, `perplexity` (which measure to use in the output results; perplexity is sentence level)
- `token-to-word`: `sum`, `average` (sum or average predictability across all sub-word tokens; sum doesn't make sense for probability)
- `roi-summary`: `macro`, `micro` (average across all words in the ROI before comparison or compare each word; for `micro` it is necessary that ROIs in both comparison settings have the same number of words)
- `k-lemmas`: number of lemmas to include in the summary; positive integer k for the top-k lemmas, negative integer k for bottom-k lemmas; `inf` to include all lemmas. 
- `punctuation`: `previous`, `next`, `separate`, `ignore` (how to handle punctation tokens; fold into previous word, next word, treat as a separate word or ignore from computation. If it is treated as separate token, this will influence ROIs.)



# Output

Saves a TSV file with the following columns: 

- `condition` (the specific condition the sentence belongs to)
- `metric` (acc, perr, ew, mw; see below)
- `mean` (average of the metric across all sentids and lemmas)
- `se` (standard error of the metric across all sentids and lemmas)

The number of rows in the output will be 4 times the number of conditions. 

## Defining metrics

- `acc`: Proportion of times Pred(expected) > Pred(unexpected)
- `perr`: $\frac{Pred(unexpected)}{Pred(expected) + Pred(unexpected))}$
- `ew`: ADD FORMULA FROM NEWMAN ET AL 
- `mw`: ADD FORMULAR FROM NEWMAN ET AL

[MAYBE IMPLEMENT JUST ONE OF EW OR MW AND HAVE STUDENTS IMPLEMENT THE OTHER PARTS?]

