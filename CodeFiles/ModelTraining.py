# c2s_trainer.py

from dataclasses import dataclass
import torch
import random
import os
import anndata
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments

# Import functions/classes from other modules
from DataPrep import vocab_gen, gen_cell_sentence, cell_type_predict_prompt
from Tokenizer import tokenize_for_training, Cell2SentenceDataCollator
from Model import load_gpt2_model

@dataclass
class Cell2SentenceConfig:
    """Configuration for Cell2Sentence training"""
    # Model parameters
    model_name_or_path: str = "gpt2"  # Base model to start from
    vocab_size: int = 50257
    max_position_embeddings: int = 1024

    # Training parameters
    output_dir: str = "./c2s_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000

    # Cell2Sentence specific parameters
    top_k_genes: int = 100  # Number of top genes to include in cell sentences
    max_eval_samples: int = 500
    loss_on_response_only: bool = True
    sentence_delimiter: str = " "

    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

class Cell2SentenceTrainer:
    """Main class for training Cell2Sentence models"""

    def __init__(self, config: Cell2SentenceConfig, model_path: str = None, tokenizer_path: str = None):
        self.config = config

        print("Loading online GPT-2 model...")
        self.tokenizer, self.model, self.device = load_gpt2_model()

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Failed to load any GPT-2 model!")

        print(f"Cell2Sentence trainer initialized!")

    def prepare_data_from_adata(self,
                               adata: anndata.AnnData,
                               task: str = "cell_type_prediction",
                               cell_type_col: str = "cell_type") -> DatasetDict:
        """ Prepare training data from AnnData object. """
        print(f"Preparing data for task: {task}")

        # Generate vocabulary and cell sentences
        vocabulary = vocab_gen(adata)
        cell_sentences = gen_cell_sentence(
            adata,
            vocabulary,
            top_k_genes=self.config.top_k_genes,
            delimiter=self.config.sentence_delimiter
        )

        # Get cell type labels
        if cell_type_col not in adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_col}' not found in adata.obs")

        cell_types = adata.obs[cell_type_col].tolist()

        # Create task-specific prompts
        if task == "cell_type_prediction":
            dataset = cell_type_predict_prompt(cell_sentences, cell_types)
        else:
            raise ValueError(f"Unsupported task: {task}")

        # Tokenize the dataset
        print("Tokenizing dataset for C2S training...")
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_for_training(
                examples,
                self.tokenizer,
                self.config.loss_on_response_only
            ),
            batched=True,
            remove_columns=dataset.column_names
        )

        # Split the dataset
        dataset_size = len(tokenized_dataset)
        train_size = int(self.config.train_split * dataset_size)
        val_size = int(self.config.val_split * dataset_size)

        indices = list(range(dataset_size))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        dataset_dict = DatasetDict({
            "train": tokenized_dataset.select(train_indices),
            "validation": tokenized_dataset.select(val_indices),
            "test": tokenized_dataset.select(test_indices) if test_indices else None
        })

        print(f"Dataset splits - Train: {len(dataset_dict['train'])}, "
              f"Val: {len(dataset_dict['validation'])}, "
              f"Test: {len(dataset_dict['test']) if dataset_dict['test'] else 0}")

        return dataset_dict

    def train(self, dataset_dict: DatasetDict):
        """ Train the Cell2Sentence model. """
        print(" Starting Cell2Sentence training...")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            dataloader_drop_last=True,
        )

        # Setup data collator
        data_collator = Cell2SentenceDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_position_embeddings
        )

        # Limit evaluation samples if specified
        eval_dataset = dataset_dict["validation"]
        if (self.config.max_eval_samples and
            len(eval_dataset) > self.config.max_eval_samples):
            eval_indices = random.sample(
                range(len(eval_dataset)),
                self.config.max_eval_samples
            )
            eval_dataset = eval_dataset.select(eval_indices)
            print(f"Limited evaluation dataset to {len(eval_dataset)} samples")

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Train the model
        print(" Training started!")
        trainer.train()

        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        print(f"Training completed! Model saved to: {self.config.output_dir}")

    def generate_text(self,
                     prompt: str,
                     max_new_tokens: int = 50,
                     temperature: float = 0.8,
                     do_sample: bool = True,
                     top_k: int = 50,
                     top_p: float = 0.9) -> str:
        """ Generate text using the trained model. """
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.replace(prompt, "").strip()

        return generated_text