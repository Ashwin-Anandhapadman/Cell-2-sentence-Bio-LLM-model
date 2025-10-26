# c2s_data_prep.py

import numpy as np
import anndata
from collections import OrderedDict
from typing import List, Dict
from tqdm import tqdm
from datasets import Dataset
import random

def vocab_gen(adata: anndata.AnnData) -> OrderedDict:
    """
    Generate vocabulary from AnnData object based on gene expression frequency.
    """
    print("Generating vocabulary from single-cell data...")

    gene_counts = {}
    for gene_idx, gene_name in enumerate(adata.var_names):
        # Count non-zero expressions for this gene
        # Assuming adata.X is an array or can be indexed
        num_expressing_cells = np.sum(adata.X[:, gene_idx] > 0)
        gene_counts[gene_name] = num_expressing_cells

    # Sort genes by expression frequency (most expressed first)
    sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
    vocabulary = OrderedDict(sorted_genes)

    print(f"Generated vocabulary with {len(vocabulary)} genes")
    return vocabulary

def gen_cell_sentence(adata: anndata.AnnData, vocabulary: OrderedDict, top_k_genes: int = 150, delimiter: str = " ") -> List[str]:
    """ 
    Generates cell sentences from sc-data.

    Args:
      adata: the AnnData object.
      vocabulary: Ordered dict of genes along with their frequencies.
      top_k_genes: top K genes to include per cell.

    Returns:
    list of cell sentences
    """
    sentences = []

   
    gene_to_rank = {gene_name: rank for rank, gene_name in enumerate(vocabulary.keys())}
    
    for cell_idx in tqdm(range(adata.n_obs), desc="Generating Sentences"):
        cell_exp = adata.X[cell_idx, :].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[cell_idx, :]
        exp_genes_idx = np.where(cell_exp > 0)[0]
        
        exp_genes = []
        for gene_idx in exp_genes_idx:
            gene_name = adata.var_names[gene_idx]
            
            # Use the pre-calculated dictionary for O(1) lookup
            if gene_name in gene_to_rank:
                vocab_rank = gene_to_rank[gene_name]
                exp_genes.append((vocab_rank, gene_name))
        
        # Sort by rank (lower rank = more frequent)
        exp_genes.sort(key=lambda x: x[0]) 
        
        top_genes = [gene_name for _, gene_name in exp_genes[:top_k_genes]]

        # Create cell sentence
        cell_sentence = delimiter.join(top_genes)
        sentences.append(cell_sentence)

    print(f"Generated {len(sentences)} cell sentences")
    return sentences

def cell_type_predict_prompt(cell_sentences: List[str], cell_types: List[str]) -> Dataset:
    """This function creates training prompts for cell type prediction"""

    prompt_templates = [
        "Given the gene expression pattern, identify the cell type.",
        "Classify the cell type based on the gene expression pattern.",
        "What is the cell type given this gene expression pattern of genes?",
    ]

    model_inputs = []
    responses = []

    for cell_sentence, cell_type in zip(cell_sentences, cell_types):
        instruct = random.choice(prompt_templates)

        model_input = f"{instruct} \n Genes: {cell_sentence} \n Cell type:"
        response = f"{cell_type}"

        model_inputs.append(model_input)
        responses.append(response)

    # Create dataset
    dataset = Dataset.from_dict({
        "model_input": model_inputs,
        "response": responses,
        "cell_sentence": cell_sentences,
        "cell_type": cell_types
    })

    print(f"Created {len(dataset)} training examples")
    return dataset