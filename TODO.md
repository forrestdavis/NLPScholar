# Short Term
- [ ] Add one flag, fp16 
- [ ] Add flag details to a Train.md 
- [ ] Check that `load_kwargs` is updated
- [ ] Update environment.yaml with the correct version of transformers

    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=True,
    )


# Mid Term
- [ ] Add ngram 
- [ ] Add custom tokenizer 
- [ ] Add compatibility with marty's neural-complexity

# Long Term
- [ ] Handle longer sequences in language models (sliding window?)
- [ ] Add hyperparameter search 
- [ ] Add whole word masking 

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
