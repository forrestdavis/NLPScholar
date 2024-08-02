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

#### exp 

There are currently 4 experiments: Interact, TSE, TextClassification, and
TokenClassification. Interact allows you to test out sentences on the command
line and see the by-word surprisal and probability values for each word. TSE
yields by-token (that is, including subwords) surprisal and probability measures
for a minimal-pair experiment (i.e., targeted syntactic evaluations).
TextClassification is for classification over texts. In other words, one label
is returned for each text inputted (e.g., sentiment analysis, natural language
inference). TokenClassification is for classification over tokens in a text. In
other words, one label is returned for each token (i.e., subword) in a text
(e.g., part of speech tagging, named entity recognition). 

#### models 

There are currently 4 types of models supported (all building on HuggingFace's
transformers library): causal language models (hf\_causal\_model), masked
language models (hf\_masked\_model), text classification models
(hf\_text\_classification\_model), and token classification models
(hf\_token\_classification\_model). You can use any model that is listed on
HuggingFace's models hub as long as they work with the relevant auto classes
(AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification,
AutoModelForTokenClassification). To refer to a model, you give it's type and
then it's model name on HuggingFace (or path to the folder on your local
computer). More than one model from each class can be specified. For example, 

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

#### other important flags 

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

#### tokenizers

#### device

#### precision 

#### showSpecialTokens

#### id2label
