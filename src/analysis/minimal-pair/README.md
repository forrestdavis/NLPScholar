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
- `sentid` (the ID of the sentences)
- `word` (the word the subword token belongs to)
- `wordpos` (the specific word in the context sentence that the subword token belongs to)
- `model` (the name of the language model)
- `tokenizer` (the name of the tokenizer)
- `punctuation` (is the token punctuation?)
- `prob` (the probability of the token given the context)
- `surp` (the surprisal of the token given the context; log base 2)




## TSV file with conditions

- `sentid` (the ID of the specific sentence; matches what is in `predictability.tsv`)
- `comparison` (expected and unexpected)
- `pairid` (the ID of the specific pair of sentences being compared)
- `contextid` (the ID of context shared across all lemmas)
- `lemma` (the word lemma)
- `condition` (the specific conditions or group that the sentence belongs to)
- `sentence` (the full sentence)
- `ROI` (the specific word positions that are of interest)

To understand the difference between sentence ID, pair ID and context ID, consider the following context from Newman et al: 

```The keys to the cabinet ____ on the table. ```

Given the context, if one wanted to compare the probability of `___` being filled with is vs. are, but also exist vs. exists, then there would be total of 4 sentences, two pairs and one context. 

All sentences for a given `model` in a given `condition` will be averaged together in the final output. There will be as many rows in the final output as there are conditions times models. For example, each of the "templates" in Marvin and Linzen (2018) and all of the "pheonomenon" in Warstadt et al (2020) can be thought of as conditions. 

In many cases, ROIs are just single words, in which case this field is filled with just one integer corresponding to the target word position. Some experiments might have multiple words of interest. In such cases, each of the words should be comma separated. Note: it is ok for the ROIs to differ across pairs. 

## Other parameters to be included in the config:

### Required parameters

- `predfpath`: File path and name for predictability TSV
- `datafpath`: File path and name for data TSV
- `resultsfpath`: File path and name for output TSV
- `pred_measure`: `probability`, `surprisal`, `perplexity` (which measure to use in the output results; perplexity is sentence level)

### Optional parameters
- `word_summary`: `sum` or `mean` predictability across all sub-word tokens. (Default is mean; Probability always uses `sum`)
- `k_lemmas`: number of lemmas to include in the summary; positive integer k for the top-k lemmas, negative integer k for bottom-k lemmas; `all` to include all lemmas (Default is `all`). 
- `punctuation`: `previous`, `next`, `separate`, `ignore` (how to handle punctation tokens; fold into previous word, next word, treat as a separate word or ignore from computation. If it is treated as separate token, this will influence ROIs.)


# Output

Saves a TSV file with the following columns: 

- `condition`
- `model`
- `microdiff`: average difference in predictability computed by comparing each individual ROI for every pair. (Will be NA for perplexity)
- `macrodiff`: average predictability across all ROIs in a sentence computed before pairwise difference is calculated. 
- `acc`: Proportion of times Pred(expected) > Pred(unexpected)

Note, the directionality for probability is different from that for surprisal and perplexity: higher probability means more predictable, whereas higher surprisal or perplexity means less predictable. 

For the difference metrics, the smaller (predicted) values are subtracted from the larger ones:
- Expected - Unexpected for probability
- Unexpected - Expected for surprisal and perplexity


