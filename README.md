# NLPScholar

This repository is used for NLP at [Colgate
University](https://www.cs.colgate.edu/), taught by Profs. [Forrest
Davis](https://forrestdavis.github.io/) and [Grusha
Prasad](https://grushaprasad.github.io/). It is mainly a toolkit for running
various NLP experiments with the aim of training NLP Scholars! It builds on top
of the wonderful [HuggingFace](https://huggingface.co/) NLP tools.

Never heard of NLP Scholars before, no worries, we wrote a paper: [Training an
NLP Scholar at a Small Liberal Arts College: A Backwards Designed Course
Proposal](https://aclanthology.org/2024.teachingnlp-1.16/) which we presented at
the [Sixth Workshop on Teaching
NLP](https://sites.google.com/view/teachingnlpacl2024) co-located with [ACL
2024](https://2024.aclweb.org/). 


**PLEASE NOTE: This is under active development. If you run into issues, we are
sorry. First thank you so much for using this. We would love to learn that you
are and how you incorporate it in your class/research. You can cite this in your
work if you want (see the bottom) :). Second, please submit an issue on github.
We will try to promptly fix it (if its not a crazy time of the semester). If you
are in our class, please email us.**  

Now onto the details!

## Environment

Install the `nlp` environment (see Install.md). Ensure you have run 

    conda activate nlp

## Running

You can run experiments via `main.py` with the relevant config file. See below
for details on the structure of the config files. 

## Basic Config Files

Interacting with the toolkit is facilitated by config files. An example one is
copied below showing how to run GPT2 in interactive mode. 

```yaml
exp: MinimalPair

mode: 
    - interact

models: 
      hf_causal_model:
          - gpt2
```

#### `exp`

There are three experiments: `MinimalPair`, `TextClassification`, and
`TokenClassification`. `MinimalPair` is for by-token (that is, including
subword) predictability measures for minimal pair experiments (i.e., targeted
syntactic evaluations). `TextClassification` is for classification over texts.
In other words, one label is returned for each text inputted (e.g., sentiment
analysis, natural language inference). `TokenClassification` is for
classification over tokens in a text.  In other words, one label is returned for
each token (i.e., subword) in a text (e.g., part of speech tagging, named entity
recognition). 

#### `mode`

Each experiment has `mode`s. There are three modes: `interact`, `train`, and
`evaluate`.  For `MinimalPair`, `interact` yields by-word (combining subword
tokens) predictability measures for an inputted sentence, `train` finetunes a
language model using the relevant language modeling objective (i.e.,
auto-regressive or masked language modeling), and `evaluate` returns by-token
predictability measures for a dataset.  For `TextClassification`, `interact`
yields by-text classification labels for inputted text, `train` finetunes a
pretrained model for text classification, and `evaluate` returns by-text labels
for a dataset.  For `TokenClassification`, `interact` yields by-token
classification labels for inputted text, `train` finetunes a pretrained model
for token classification, and `evaluate` returns by-token classification labels
for a dataset. 

#### `models`

There are 4 types of models supported (all building on HuggingFace's
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

## Config Details for `evaluate`

#### `condfpath` and `predfpath`

In running in `evaluate` mode, you need to specify the path to the file with the
data you will be running the model on (the `condfpath`) and the path to where
you want to save the predictions (the `predfpath`). For example, 

```yaml
exp: TSE

models: 
      hf_causal_model:
          - gpt2
      hf_masked_model:
          - bert-base-cased

condfpath: conditions/minimal_pairs.tsv

predfpath: predictions/minimal_pairs.tsv
```

Runs GPT2 and BERT on the minimal pairs in `conditions/minimal_pairs` and saves the
output to `predictions/minimal_pairs.tsv`.

#### `checkFileColumns`

In evaluating models (in `evaluate` mode), you can check that the necessary
columns for the broader experiment are included in the file indexed with
`condfpath`. The default value is `True`, meaning the columns are checked. 

```yaml
checkFileColumns: True
```

#### `loadAll`

In evaluating models (in `evaluate` mode), more than one model can be evaluated.
`loadAll` controls memory usage, with `True` for loading all models to memory at
once and `False` for loading one model at a time into memory. The default
behavior is `False`. 

```yaml
loadAll: False
```

## Config Details for `train`

#### `trainfpath`, `validfpath`, and `modelfpath`

In `train` mode you need to specify training data, validation data, and an
output directory for the final model. Training and validation data can either be
from HuggingFace's dataset options on their hub or local json or tsv files. To
specify a HuggingFace remote dataset specify the name, task (if applicable), and
split with colon seperators. See below for two examples, one loading 
[imdb data](https://huggingface.co/datasets/stanfordnlp/imdb), which doesn't
have subtasks: 

```yaml
trainfpath: imdb:train
validfpath: imdb:test
modelfpath: imdb_model
```

and one loading mnli from [glue](https://huggingface.co/datasets/nyu-mll/glue): 

```yaml
trainfpath: nyu-mll/glue:mnli:train
validfpath: nyu-mll/glue:mnli:validation_matched
modelfpath: mnli_model
```

#### `loadPretrained`

You can specify if you want to load the pretrained weights of a model or
randomly initialize a model with the same architecture as the model name with
`loadPretrained`. If set to `False` and training for classification you must
specify the number of labels with `numLabels`. The default is `True`. 

```yaml
loadPretrained: True
```

#### `numLabels`

You can specify the number of classification labels for token or text
classification with `numLabels`. Note: you must specify a value if
`loadPretrained` is `False`. 

```yaml
numLabels: 5
```

#### `maxSequenceLength`

In loading a model, you can specify the maximum sequence length (i.e., the
context size) with `maxSequenceLength`. This changes the sequence length of the
model when loading not from a pretrained model and controls the sequence length
in a batch during training. The default is 128. 

```yaml
maxSequenceLength: 128
```

#### `seed`

You can specify the seed for shuffling the dataset initially with `seed`. The
default value is 23.

```yaml
seed: 23
```

#### `samplePercent`

It is often helpful to run training code with a subset of your data (e.g., for
debugging). You can specify what percent of your data to use with
`samplePercent`. This can either be a whole number between 0 and 100 (which is
converted to a percent; for example, 10 translates to 0.10), or a float (e.g.,
0.001). The default is None, which results in no sampling (i.e., all data is
used). 

```yaml
samplePercent: 10
```

#### `textLabel`

For training either with language model objectives or text/token classification,
your data needs to point to the text. The column with this information is
specified with `textLabel`. The default is `"text"` which means your text data
for training should be labeled with a column with that name. 

```yaml
textLabel: text
```

#### `pairLabel`

For text classification, you can specify two sentences for tasks like natural
language inference and paraphrase detection. You can specify the column with the
second sentence using `pairLabel`. The default is `"pair"`. The example below
shows how to specify the correct columns for glue's mnli task. 

```yaml
textLabel: premise
pairLabel: hypothesis
```

#### `tokensLabel`

For token classification, the dataset must also provide the tokens (e.g., the
words). You can specify the column with this data with `tokensLabel`. The
default is `"tokens"`. 

```yaml
tokensLabel: tokens
```

#### `tagsLabel`

For token classification, the dataset must also provide the per-token tags
(e.g., the named-entity tags). You can specify the column with this data with
`tagsLabel`.  The default is `"tags"`. 

```yaml
tagsLabel: tags
```

### Extra Training Settings

You can specify the following additional training settings using the config file
(lightly adapted from HuggingFace's Trainer arguments and [Trainer
class](https://huggingface.co/docs/transformers/main_classes/trainer#trainer)):

    modelfpath: The output directory where the model is saved
    epochs: Number of training epochs. The default is 2. 
    eval_strategy: The evaluation strategy to use ('no' | 'steps'
                    | 'epoch'). The default is 'epoch'.
    eval_steps: Number of update steps between two evaluations.
                        The default is 500. 
    batchSize: The per-device batch size for train/eval. The
                        default is 8. 
    learning_rate: The initial learning rate for AdamW. Default
                            is 5e-5. 
    weight_decay: Weight decay. The default is 0.01.
    save_strategy: The checkpoint save strategy to use ('no'
                    | 'steps' | 'epoch'). The default is 'epoch'.
    save_steps: Number of update steps between saves. Default is
                        500.
    load_best_model_at_end: Whether or not to load the best model
                    at the end of training. If True, the best model will
                    be saved as the model at the end. Default is False.
    maskProbability: The rate of dynamic masking for masked
                    language modeling. The default is 0.15. 

## Additional Config Settings

There are further parameters that can be specified, detailed below. 

#### `device`

You can specify the device you want to run the model on with `device`: 

```yaml
device: mps
```

The options are `best`, `mps`, `cuda`, or `cpu` (specific devices can also be
specified). The default is `best`, which prioritizes `mps` and then `cpu`. (You
must specify `cuda` or similar to use GPUs). 

#### `precision`

You can control the memory requirements for your experiments with `precision`,
which controls the precision of the loaded model (if applicable): 

```yaml
precision: 16bit
```

 The options are `full`, `16bit`, `8bit`, and `4bit`. The default behavior is
`full`, which loads the model without changing its precision. Selecting `16bit`
with `train` will train a lower precision model (note: you need to use a GPU for
this). 

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

#### `batchSize`

In experiments, you can control the batch size of the model with `batchSize`.
The default size is 1 in `evaluate` mode and 16 in `train` mode.

```yaml 
batchSize: 1 
```

#### `verbose`

In running experiments, you can control verbosity with `verbose`. When set to
`True`, it prints some more information to the screen.  The default behavior is
`True`. 

```yaml
verbose: True
```

## Citation

Please cite our NLP Scholar paper: 

```bibtex
@inproceedings{prasad-davis-2024-training-nlp,
  title = {Training an {NLP} Scholar at a Small Liberal Arts College: A Backwards Designed Course Proposal},
  author = {Prasad, Grusha and Davis, Forrest},
  editor = {Al-azzawi, Sana and Biester, Laura and Kov{\'a}cs, Gy{\"o}rgy and Marasovi{\'c}, Ana and Mathur, Leena and Mieskes, Margot and Weissweiler, Leonie},
  booktitle = {Proceedings of the Sixth Workshop on Teaching NLP},
  month = aug,
  year = {2024},
  address = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2024.teachingnlp-1.16},
  pages = {105--118},
}
```

This wouldn't work without HuggingFace! Please cite them too: 

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}

@inproceedings{lhoest-etal-2021-datasets,
    title = "Datasets: A Community Library for Natural Language Processing",
    author = "Lhoest, Quentin  and
      Villanova del Moral, Albert  and
      Jernite, Yacine  and
      Thakur, Abhishek  and
      von Platen, Patrick  and
      Patil, Suraj  and
      Chaumond, Julien  and
      Drame, Mariama  and
      Plu, Julien  and
      Tunstall, Lewis  and
      Davison, Joe  and
      {\v{S}}a{\v{s}}ko, Mario  and
      Chhablani, Gunjan  and
      Malik, Bhavitvya  and
      Brandeis, Simon  and
      Le Scao, Teven  and
      Sanh, Victor  and
      Xu, Canwen  and
      Patry, Nicolas  and
      McMillan-Major, Angelina  and
      Schmid, Philipp  and
      Gugger, Sylvain  and
      Delangue, Cl{\'e}ment  and
      Matussi{\`e}re, Th{\'e}o  and
      Debut, Lysandre  and
      Bekman, Stas  and
      Cistac, Pierric  and
      Goehringer, Thibault  and
      Mustar, Victor  and
      Lagunas, Fran{\c{c}}ois  and
      Rush, Alexander  and
      Wolf, Thomas",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-demo.21",
    pages = "175--184",
    abstract = "The scale, variety, and quantity of publicly-available NLP datasets has grown rapidly as researchers propose new tasks, larger models, and novel benchmarks. Datasets is a community library for contemporary NLP designed to support this ecosystem. Datasets aims to standardize end-user interfaces, versioning, and documentation, while providing a lightweight front-end that behaves similarly for small datasets as for internet-scale corpora. The design of the library incorporates a distributed, community-driven approach to adding datasets and documenting usage. After a year of development, the library now includes more than 650 unique datasets, has more than 250 contributors, and has helped support a variety of novel cross-dataset research projects and shared tasks. The library is available at https://github.com/huggingface/datasets.",
    eprint={2109.02846},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
}

```

If you use Kauf and Ivanova's PPL scoring technique, please cite them: 

```bibtex
@inproceedings{kauf-ivanova-2023-better,
    title = "A Better Way to Do Masked Language Model Scoring",
    author = "Kauf, Carina  and
      Ivanova, Anna",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.80",
    doi = "10.18653/v1/2023.acl-short.80",
    pages = "925--935",
    abstract = "Estimating the log-likelihood of a given sentence under an autoregressive language model is straightforward: one can simply apply the chain rule and sum the log-likelihood values for each successive token. However, for masked language models (MLMs), there is no direct way to estimate the log-likelihood of a sentence. To address this issue, Salazar et al. (2020) propose to estimate sentence pseudo-log-likelihood (PLL) scores, computed by successively masking each sentence token, retrieving its score using the rest of the sentence as context, and summing the resulting values. Here, we demonstrate that the original PLL method yields inflated scores for out-of-vocabulary words and propose an adapted metric, in which we mask not only the target token, but also all within-word tokens to the right of the target. We show that our adapted metric (PLL-word-l2r) outperforms both the original PLL metric and a PLL metric in which all within-word tokens are masked. In particular, it better satisfies theoretical desiderata and better correlates with scores from autoregressive models. Finally, we show that the choice of metric affects even tightly controlled, minimal pair evaluation benchmarks (such as BLiMP), underscoring the importance of selecting an appropriate scoring metric for evaluating MLM properties.",
}
```

## Other Resources

If you like this toolkit and want more tools or different tools, check out the
following repositories: 

- Kanishka Misra's wonderful
  [minicons](https://github.com/kanishkamisra/minicons)
