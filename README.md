# CELL-2-Sentence bio LLM model

This repo is about re-building the popular bio LLM model: Cell2Sentence (C2S) from scratch. Here, a LLM model is fine-tuned using scRNA seq (processed) data to enable cell type prediction.


This code provides flexibility to run synthetically generated data, as well as real scRNA seq data. Due to compute constraints I have tested the model with GPT2. Future updates to this repo will include testing with bigger and better LLM models. 

## Overview of the model:
1. The model designed here is inspired from the C2S model. The model generates cell sentences (for each cell in scRNA seq data) which is basically a sorted list of genes expressed in the cell (based on their expression frequency across cells).
2. The generated cell sentences are then used for training the model to enable cell type prediction.
3. Once trained, the model will be able to predict cell type if given a sorted list of genes.
4. Training details:
    a. Each cell sentence is clubbed with a prompt for LLM and then this combined prompt + response (sequence) is then tokenized using the Transformers library. 
    b. The tokenization process also gives the flexibility to hide the prompt section of the tokens, so that the model only learns to predict cell type (response) section of the prompt + response pair.
    c. Rest of the training parameters are the default parameters given in literature/model card.
    d. Hugging Face's "trainer" library handles the training process. We then use validation data to check response.
    e. The code is also configured to run a real scRNA seq data (like PBMC 3K) and predict cell types. 


Original C2S paper: https://www.biorxiv.org/content/10.1101/2023.09.11.557287v1.full.pdf