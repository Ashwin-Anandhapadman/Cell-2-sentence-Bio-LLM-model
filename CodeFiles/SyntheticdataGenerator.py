# c2s_demo_data.py

import numpy as np
import anndata
import scanpy as sc
import os
from typing import List

def create_synthetic_data(n_cells: int = 1000, n_genes: int = 500) -> anndata.AnnData:
    """ Create synthetic single-cell data for demonstration. """
    print(f" Creating synthetic data: {n_cells} cells × {n_genes} genes")

    np.random.seed(42)
    expression_data = np.random.lognormal(0, 1, (n_cells, n_genes))
    expression_data = np.where(expression_data > 2, expression_data, 0)  # Add sparsity

    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    base_types = ["T_cell", "B_cell", "NK_cell", "Monocyte"]
    cell_types = (base_types * ((n_cells // len(base_types)) + 1))[:n_cells]

    adata = anndata.AnnData(
        X=expression_data,
        obs={"cell_type": cell_types},
        var={"gene_names": gene_names}
    )
    adata.var_names = gene_names
    adata.obs_names = [f"Cell_{i:04d}" for i in range(n_cells)]

    print(f"Synthetic dataset created!")
    return adata

def load_real_adata(data_file: str) -> anndata.AnnData:
    """
    Load real AnnData file and prepare it for Cell2Sentence training.
    """
    print(f"Loading real data from: {data_file}")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    adata = sc.read_h5ad(data_file)

    print(f" Data loaded successfully! Shape: {adata.n_obs} cells × {adata.n_vars} genes")

    potential_cell_type_cols = [
        'cell_type', 'celltype', 'cell_ontology_class',
        'annotation', 'cluster', 'leiden', 'seurat_clusters'
    ]

    available_cell_type_cols = [col for col in potential_cell_type_cols if col in adata.obs.columns]

    if available_cell_type_cols:
        print(f" Available cell type columns: {available_cell_type_cols}")
    else:
        print("No obvious cell type column found.")

    return adata