# Basic trainer child class for language modeling
# building on HuggingFace models
# Implemented by Forrest Davis 
# (https://github.com/forrestdavis)
# August 2024
from .Trainer import Trainer
import datasets
import transformers 
import sys
import math

class HFLanguageModelTrainer(Trainer): 

    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)


    def tokenize_function(self, examples):
        result = self.Model.tokenizer(examples[self.textLabel])
        if self.wholeWordMasking:
            result['word_ids'] = [result.word_ids(i) 
                                  for i in range(len(result["input_ids"]))]
        return result

    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in
                                examples.keys()}

        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Split by chunks of maxSequenceLength
        result = {
            k: [t[i : i + self.maxSequenceLength] for i in range(0,
                                                                 total_length,
                                                                 self.maxSequenceLength)]
            for k, t in concatenated_examples.items()
        }
        return result

    def preprocess_dataset(self):
        if self.verbose:
            sys.stderr.write("Tokenizing the dataset...\n")
        self.dataset = self.dataset.map(self.tokenize_function, batched=True, 
                                       remove_columns=[self.textLabel])
        if self.verbose:
            sys.stderr.write("Chunking the dataset...\n")
        self.dataset = self.dataset.map(self.group_texts, batched=True)
        self.data_collator = \
                    transformers.DataCollatorForLanguageModeling(
                        tokenizer=self.Model.tokenizer._tokenizer, 
                    mlm=self.Model.isMaskedModel, 
                    mlm_probability=self.maskProbability)

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

        if self.precision == '16bit':
            fp16 = True
        else:
            fp16 = False

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
            fp16=fp16,
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
        eval_results = trainer.evaluate()
        print(f"Perplexity on evaluation data: " \
              f"{math.exp(eval_results['eval_loss']):.2f}")
        trainer.save_model()

