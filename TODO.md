# Short Term
- [ ] Add in LanguageModel experiment with evaluate being perplexity, train
  being like MinimalPair, and interact being ppl for inputted text
- [ ] Handle longer sequences in language models (sliding window?)
  (https://huggingface.co/docs/transformers/en/perplexity)
      problem is currently pad token is being used in loss calculation (plus
    CLS/SEP for masked)
- [ ] refactor `get_output` to `get_logits` 
- [ ] add in `get_perplexity` 
- [ ] should i factor out the sliding window loop? 
- [ ] put the masking in a function to allow for additions
- [ ] rename by sentence perplexity to by batch ?

# Mid Term
- [ ] Add ngram 
- [ ] Add custom tokenizer 
- [ ] Add compatibility with marty's neural-complexity
- [ ] How do we specify model details for training model from scratch, should we
    have a build? How would this interact with usePretrained? Could instead have a
    pointer to a config file with the relevant details for RNNs, like a HuggingFace
    config? 
- [ ] refactor named tuple as dataclass

# Long Term
- [ ] Add hyperparameter search 
- [ ] Add whole word masking 
- [ ] Add in generation experiments
- [ ] Website 
- [ ] Package 

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
