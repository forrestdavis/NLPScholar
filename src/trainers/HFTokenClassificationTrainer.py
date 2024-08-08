from .Trainer import Trainer
import datasets
import transformers 
import sys

class HFTextClassificationTrainer(Trainer): 

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)
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
    def train(self):

        # Set up raw dataset
        self.set_dataset()

        # Preprocess_dataset
        self.preprocess_dataset()

        # Shuffle
        self.dataset = self.dataset.shuffle(seed=42)

        use_cpu = False
        if self.device == 'cpu': 
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
            )
        trainer.train()
        if self.verbose: 
            sys.stderr.write(f"Saving final model to {self.modelfpath}...\n")
        trainer.save_model()
