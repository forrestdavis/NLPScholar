# Short Term
- [ ] Add in LanguageModel experiment with evaluate being perplexity, train
  being like MinimalPair
- [ ] Add documentation for LanguageModel
- [ ] Add explanation of analysis to README
- [ ] Set up config files for analysis

# Mid Term
- [ ] Add ngram 
- [ ] Add custom tokenizer 
- [ ] Add compatibility with marty's neural-complexity
- [ ] Setup evaluate to run on datasets from Huggingface
- [ ] Add a method to use pretrained tf models 

# Long Term
- [ ] Add hyperparameter search 
- [ ] Add whole word masking
- [ ] Some tutorials 

# Dreams 
- [ ] Make documentation page
- [ ] Make a package on pip
- [ ] Add unit tests 
- [ ] Make PPL calculation for MLM's faster. Right now, there is additional time
  complexity added by the token-wise masking. If this could be vectorized and
the batching reoriented smartly maybe it would speed things up? 

# Done 

- [X] Add token classification trainer 
- [X] Add text classification trainer 
- [X] rename experiment to evaluate
- [X] rename TSE to MinimalPair
- [X] add mode flag 
- [X] interactive mode with classification
- [X] Handle longer sequences in classification (truncate)
- [X] Add in pretrained vs. new model loading
- [X] Add language model trainer 
- [X] Add one flag, fp16 
- [X] Add flag details to a README.md 
- [X] Check that `load_kwargs` is updated
- [X] Update environment.yaml with the correct version of transformers
- [X] Handle longer sequences in language models (sliding window?)
- [X] Add in LanguageModel interact being ppl for inputted text
