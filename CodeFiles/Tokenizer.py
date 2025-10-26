# c2s_tokenizer.py

from dataclasses import dataclass
from typing import Dict
import torch
from transformers import GPT2Tokenizer

def tokenize_for_training(examples: Dict, tokenizer, loss_on_response_only: bool = True):
    """
    Tokenize examples for training with optional response-only loss.
    """
    batch_size = len(examples["model_input"])
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for i in range(batch_size):
        model_input = examples["model_input"][i]
        response = examples["response"][i]
        full_text = model_input + response

        full_tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None
        )

        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]

        if loss_on_response_only:
            input_tokenized = tokenizer(
                model_input,
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding=False,
                return_tensors=None
            )
            input_length = len(input_tokenized["input_ids"])

            # Create labels: -100 for input tokens (ignored), actual token ids for response
            labels = [-100] * input_length + input_ids[input_length:]
            labels = labels[:len(input_ids)]  # Ensure same length
        else:
            labels = input_ids.copy()

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

@dataclass
class Cell2SentenceDataCollator:
    """Custom data collator for Cell2Sentence training."""

    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        # Find max length in batch and cap it
        max_length = max(len(ex["input_ids"]) for ex in examples)
        max_length = min(max_length, self.max_length)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for example in examples:
            input_ids = example["input_ids"][:max_length]
            attention_mask = example["attention_mask"][:max_length]
            labels = example["labels"][:max_length]

            # Pad sequences (left padding for GPT-style models)
            padding_length = max_length - len(input_ids)

            input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + attention_mask
            labels = [-100] * padding_length + labels

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }