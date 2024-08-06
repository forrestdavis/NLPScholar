
from .Trainers import Trainer
import datasets
import transformers 
import evaluate
import numpy as np
import sys
import torch

class TextClassificationTrainer(Trainer): 

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)
        self.preprocess_dataset()
        self.evaluator = evaluate.load("accuracy")

    def preprocess_function(self, examples):
        return self.Model.tokenizer(examples['text'], truncation=True)

    def preprocess_dataset(self):
        if self.verbose:
            sys.stderr.write("Tokenizing the dataset...\n")
        self.dataset = self.dataset.map(self.preprocess_function, batched=True)
        self.data_collator = \
                    transformers.DataCollatorWithPadding(
                        tokenizer=self.Model.tokenizer._tokenizer)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = torch.argmax(predictions, axis=-1)
        results = self.evaluator.compute(predictions=predictions, references=labels)
        return {'accuracy': results['accuracy']}

    def train(self):
        training_args = transformers.TrainingArguments(
            output_dir="imdb_model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            )
        trainer = transformers.Trainer(
                model = self.Model.model, 
                args=training_args, 
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['valid'], 
                tokenizer=self.Model.tokenizer._tokenizer, 
                data_collator = self.data_collator, 
                compute_metrics = self.compute_metrics,
            )
        trainer.train()
