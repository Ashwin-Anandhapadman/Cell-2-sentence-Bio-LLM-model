# c2s_model.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

def load_gpt2_model():
    """
    Load GPT-2 model from Hugging Face.

    Returns:
        Tuple of (tokenizer, model, device)
    """
    try:
        print("Loading GPT-2 from Hugging Face...")

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = GPT2LMHeadModel.from_pretrained("gpt2")

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)

        print(f"GPT-2 loaded! Parameters: {model.num_parameters():,}")
        print(f"Device: {device}")

        return tokenizer, model, device

    except Exception as e:
        print(f" Failed to load online model: {str(e)}")
        return None, None, None