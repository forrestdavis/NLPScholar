# Defining classification analysis

The toolkit supports analysis of two types of classification tasks: 

- Encoder classification: Tasks where the model classifies the entire sequence.
- Decoder classification: Tasks where the model classifies each of the tokens in the sequence. 

# Input requirements

## TSV file with model output

This has one input-output pair per line. In encoder only classification, the sequences are the inputs, whereas in decoder classification the tokens in the sequences are inputs. 

 The tsv has the following columns: 

- `inputid` (the ID of the specific input being classified)
- `sentid` (the ID of the specific sentence used for input; in encoder clasification this is the same as inputid)
- `target` (the correct classification label)
- `predicted` (the predicted classification label)

## TSV file with conditions

WHAT IS THE FORMAT THAT WOULD BE HELPFUL HERE FOR DECODER CLASSIFICATION? A STRING WITH THE ENTIRE SEQUENCE? SOMETHING ELSE?

- `inputid` (the ID of the specific input being classified) (maybe a sequence for decoder only?)
- `input` (the sequence being classified)
- `condition` (the condition that the input belongs to) 
- `target` (the target label for the input) (maybe a sequence for decoder only?)

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




