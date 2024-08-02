# Defining classification analysis

The toolkit supports analysis of two types of classification tasks: 

- Encoder classification: Tasks where the model classifies the entire sequence.
- Decoder classification: Tasks where the model classifies each of the tokens in the sequence. 

Following the terminology used by HuggingFace, our toolkit supports two types of
classification tasks:

- Text classification: Tasks where the model assigns a label to the entire
  sequence.
- Token classification: Tasks where the model assigns a label to each token in a
  sequence. 

# Input requirements

## TSV file with model output

For text (sequence) classification, the prediction is for the whole text. The
output has the following columns: 

- `textid`: the ID of the specific pair of text being compared;
            matches what is in `predictability.tsv`
- `target`: the target label that this text belongs to
- `model`: the name of the classifier
- `tokenizer`: the name of the tokenizer
- `predicted`: the predicted label for this text belong
- `prob`: the predicted label's probability

For token classification, the prediction is for each token. The output has the
following columns:

- `token`: the subword token
- `textid`: the ID of the specific pair of text being compared;
            matches what is in `predictability.tsv`
- `word`: the word the subword token belongs to
- `wordpos`: the specific word in the context sentence that the subword token belongs to
- `condition`: the specific condition, e.g., "grammatical" or "ungrammatical"
- `model`: the name of the classifier
- `tokenizer`: the name of the tokenizer
- `punctuation`: is the token punctuation?
- `target`: the target label that this text belongs to (or None if none
            was specified)
- `predicted`: the predicted label for this text belong
- `prob`: the probability of the token given the context

## TSV file with conditions

- `textid`: the ID of the specific pair of text being compared;
            matches what is in `predictability.tsv`
- `text`: the full text
- `pair`: *optional* second sentence to use as a pair 
- `condition`: the specific conditions or group that the text
            belongs to
- `target`: the target label or labels (space separated) that this text belongs to
            Note: this assume you have one per word for token level predictions,
            if you are missing words, the labels will be aligned left-to-right
            with None as the padded target. 

The toolkit assumes that all of the possible tasks classes are seen in the evaluation data. 

## Other parameters inputted with flags:

- `predfpath`: File path and name for model predictions TSV
- `condfpath`: File path and name for conditons TSV
- `outfpath`: File path and name for output TSV
- `classification-type`: encoder, decoder


# Output

Saves a TSV file with the following columns: 

- `condition` (the specific condition the sentence belongs to)
- `class` (the specific class an input could belong to)
- `accuracy` (average accuracy over all inputs for specific class and condition combination)
- `precision` (average precision over all inputs for specific class and condition combination)
- `recall` (average recall over all inputs for specific class and condition combination)


