# NLPToolKit

## Environment

Install the `nlp` environment (see Install.md). Ensure you have run 

    conda activate nlp

## Running Models

You can run experiments via `main.py` with the relevant config file. See below
for details on the structure of the config files. 

### Basic Config Files

Interacting with the toolkit is facilitated by config files. An example one is
copied below showing how to run GPT2 in interactive mode. 

```yaml
exp: Interact

models: 
      hf_causal_model:
          - gpt2
```

#### `exp`

There are currently 4 experiments: `Interact`, `TSE`, `TextClassification`, and
`TokenClassification`. `Interact` allows you to test out sentences on the
command line and see the by-word surprisal and probability values for each word.
`TSE` yields by-token (that is, including subwords) surprisal and probability
measures for a minimal-pair experiment (i.e., targeted syntactic evaluations).
`TextClassification` is for classification over texts. In other words, one label
is returned for each text inputted (e.g., sentiment analysis, natural language
inference). `TokenClassification` is for classification over tokens in a text.
In other words, one label is returned for each token (i.e., subword) in a text
(e.g., part of speech tagging, named entity recognition). 

#### `models`

There are currently 4 types of models supported (all building on HuggingFace's
transformers library): causal language models (`hf_causal_model`), masked
language models (`hf_masked_model`), text classification models
(`hf_text_classification_model`), and token classification models
(`hf_token_classification_model`). You can use any model that is listed on
HuggingFace's models hub as long as they work with the relevant auto classes
(`AutoModelForCausalLM`, `AutoModelForMaskedLM`,
`AutoModelForSequenceClassification`, `AutoModelForTokenClassification`). To
refer to a model, you give it's type and then it's model name on HuggingFace (or
path to the folder on your local computer). More than one model from each class
can be specified. For example, 

```yaml
models: 
      hf_causal_model:
          - gpt2
      hf_masked_model:
          - bert-base-cased
          - roberta-base
```

Will load one causal language model (GPT2) and two masked language models (BERT
and RoBERTa).

#### other important options

In running the non-interactive experiments, you need to specify the path to the
file with the data you will be running the model on (the `condfpath`) and the
path to where you want to save the predictions (the `predfpath`). For example, 

```yaml
exp: TSE

models: 
      hf_causal_model:
          - gpt2
      hf_masked_model:
          - bert-base-cased

condfpath: stimuli/minimal_pairs.tsv

predfpath: results/minimal_pairs.tsv
```

Runs GPT2 and BERT on the minimal pairs in `stimuli/minimal_pairs` and saves the
output to `results/minimal_pairs.tsv`.

### Additional Details 

There are further parameters that can be specified, detailed below. 

#### `device`

You can specify the device you want to run the model on with `device`: 

```yaml
device: mps
```

The options are `best`, `mps`, `cuda`, or `cpu` (specific devices can also be
specified). The default is `best`, which prioritizes `cuda`, then `mps`, then
`cpu`. 

#### `precision`

You can control the memory requirements for your experiments with `precision`,
which controls the precision of the loaded model (if applicable): 

```yaml
precision: 16bit
```

 The options are `full`, `16bit`, `8bit`, and `4bit`. The default behavior is
`full`, which loads the model without changing its precision. 

#### `showSpecialTokens`

You can surface special tokens in the output using show `showSpecialTokens`,
which takes a boolean: 

```yaml
showSpecialTokens: True
```

If set to `True`, the model returns non-pad special tokens used internal to the
model architecture (e.g., [CLS]).

#### `PLL_type`

```yaml
PLL_type: original
```

For masked language models, the predictability measures are gathered by
iteratively masking each token in the input. Following, [Kauf and Ivanova
(2023)](https://aclanthology.org/2023.acl-short.80/), this 'pseudo-likelihood'
can be determined via different masking schemes. Presently we have two options,
`original` which simply masks each token in the input one by one, following
[Salazar et al.  (2020)](https://aclanthology.org/2020.acl-main.240/), and
`within_word_l2r` which handles words that are subworded by masking each subword
token to the right of the word being predicted. The default is `within_word_l2r`
as Kauf and Ivanova find it performs better. 

#### `id2label`

For classification models, it can be helpful to specify a mapping from model
labels to identifiable. You can do this by specifying the mappings, as below: 

```yaml
id2label:
    1: Positive
    0: Negative
```

This maps the output index 1 to Positive and 0 to Negative. The default behavior
is to use the models id2label attribute. 

#### `tokenizers`

You can specify the specific tokenizers you want to use with `tokenizers`:

```yaml
models: 
        hf_causal_model:
            - gpt2-medium
tokenizers:
        hf_tokenizer: 
            - gpt2
```

At the moment, the only supported tokenizers are those using HuggingFace's
`AutoTokenizer` class. Note that the tokenizers and models are aligned lazily.
The assumption is that for each model listed in models, the tokenizer is in the
same position in the tokenizer list. For example,  

```yaml
models: 
        hf_causal_model:
            - gpt2
        hf_masked_model:
            - bert-base-uncased
tokenizers:
        hf_tokenizer: 
            - bert-base-uncased
            - gpt2
```

Loads GPT2 with BERT's tokenizer and BERT with GPT2's tokenizer. The default
behavior is to load the version of the tokenizer associated with the model. The
default behavior is strongly encouraged. 

#### `doLower`

The input text can be lowercased with `doLower`, which takes a boolean. If set
to `True` the text is lowercased (with special tokens preserved). The default
behavior is `False`. Note this does not override the tokenizer's default
behavior. For example, if you do not set `doLower` to `True` but use
`bert-base-uncased`, the text will still be lowercased. 

```yaml
doLower: True
```
#### `addPrefixSpace`

Some tokenizers include white space in their tokens (e.g., Byte-based tokenizers
as in GPT2). For these tokenizers, the presence of a initial space affects
tokenization. You can explicitly set this behavior with `addPrefixSpace`, which
will add a prefix space if set to `True`. The default behavior is to not add a
space (i.e., `False`), following other libraries like minicons. 

```yaml
addPrefixSpace: True
```

#### `addPadToken`

For batched input, it is important to have a pad token set. This can be done
with `addPadToken`. You can either put `True` or `eos_token` if you want to use
the model's eos token or give a specific token (e.g., PAD). Note, only a word
with a single token id (i.e., a word not sub-worded) can be used as a pad token.
An error will be thrown if this is not the case. Note, this will not override
the model's pad token if one already exists. The default behavior is to use
eos, if no pad token exists. If none is provided and their is no eos token in
the model, an error may be thrown during tokenization. 

```yaml
addPadToken: True
```

#### loadAll

#### batchSize

#### checkFileFormat

#### verbose
