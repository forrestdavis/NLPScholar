# Basic trainer child class for token classification
# building on HuggingFace models
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024
from .Trainer import Trainer
import datasets
import transformers 
import sys

import numpy as np
import evaluate

# Load metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load('precision')
recall = evaluate.load('recall')
f1 = evaluate.load('f1')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        for p, l in zip(prediction, label):
            if l != -100:
                true_predictions.append(p)
                true_labels.append(l)
    predictions = true_predictions
    labels = true_labels
    acc = accuracy.compute(predictions=predictions, references=labels)
    p = precision.compute(predictions=predictions, 
                                    references=labels,
                                    average='weighted', 
                                    zero_division=0)
    r = recall.compute(predictions=predictions, 
                                    references=labels,
                                    average='weighted', 
                                    zero_division=0)
    f = f1.compute(predictions=predictions, 
                                    references=labels,
                                    average='weighted')
    return {**acc, **p, **r, **f}

class HFTokenClassificationTrainer(Trainer): 

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)

    def preprocess_function(self, examples):
        """ Tokenizes input and aligns tokens with token level labels accounting
        for split words. """
        # Adapted from HuggingFace's Token Classification Guide
        tokenized_inputs = self.Model.tokenizer(examples[self.tokensLabel],
                                                truncation=True,
                                                is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[self.tagsLabel]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    def preprocess_dataset(self):
        if self.verbose:
            sys.stderr.write("Tokenizing the dataset...\n")
        self.dataset = self.dataset.map(self.preprocess_function, batched=True)
        self.data_collator = \
                    transformers.DataCollatorForTokenClassification(
                        tokenizer=self.Model.tokenizer._tokenizer)

    def train(self):

        if self.dataset is None:
            # Set up raw dataset
            self.set_dataset()

        if 'input_ids' not in self.dataset['train'].features:
            # Preprocess_dataset
            self.preprocess_dataset()

        # Shuffle
        self.dataset = self.dataset.shuffle(seed=42)

        use_cpu = False
        if self.Model.device == 'cpu': 
            use_cpu = True

        training_args = transformers.TrainingArguments(
            output_dir=self.modelfpath,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batchSize,
            per_device_eval_batch_size=self.batchSize,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            load_best_model_at_end=self.load_best_model_at_end,
            use_cpu=use_cpu,
            )

        trainer = transformers.Trainer(
                model = self.Model.model, 
                args=training_args, 
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['valid'],
                tokenizer=self.Model.tokenizer._tokenizer, 
                data_collator = self.data_collator, 
                compute_metrics = compute_metrics,
            )
        trainer.train()
        if self.verbose: 
            sys.stderr.write(f"Saving final model to {self.modelfpath}...\n")
        trainer.save_model()
