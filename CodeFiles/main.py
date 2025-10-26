# main.py

import argparse
import sys
import os

# Import modules
from ModelTraining import Cell2SentenceConfig, Cell2SentenceTrainer
from SyntheticdataGenerator import create_synthetic_data, load_real_adata

def run_demo(model_path: str = None, tokenizer_path: str = None, train_model: bool = False, data_file: str = None):
    """ Orchestrates the Cell2Sentence demonstration flow. """
    print("Cell2Sentence Demo")
    print("=" * 50)

    # Configuration
    config = Cell2SentenceConfig(
        model_name_or_path=model_path or "gpt2",
        output_dir="./cell2sentence_demo",
        num_train_epochs=1 if train_model else 1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        top_k_genes=50,
        max_eval_samples=100,
        learning_rate=5e-5,
    )

    # Load data - real or synthetic
    if data_file:
        adata = load_real_adata(data_file)
        
        # Detect cell type column
        potential_cols = ['cell_type', 'celltype', 'cell_ontology_class', 'annotation', 'cluster', 'leiden']
        cell_type_col = None

        for col in potential_cols:
            if col in adata.obs.columns:
                cell_type_col = col
                print(f" Using '{col}' as cell type column")
                break

        if not cell_type_col:
            print("No cell type column found, using first available column")
            cell_type_col = adata.obs.columns[0]
    else:
        adata = create_synthetic_data(n_cells=200, n_genes=100)
        cell_type_col = "cell_type"

    # Initialize trainer
    trainer = Cell2SentenceTrainer(config, model_path, tokenizer_path)

    # Prepare data
    dataset_dict = trainer.prepare_data_from_adata(
        adata,
        task="cell_type_prediction",
        cell_type_col=cell_type_col
    )

    # Train model (optional)
    if train_model:
        print("  Starting training - this may take a while...")
        trainer.train(dataset_dict)
    else:
        print("⏭ Skipping training (set train_model=True to train)")

    # Test generation
    print("\nTesting Cell2Sentence generation...")
    test_prompts = [
        "Given the following gene expression pattern, predict the cell type:\nGenes: Gene_001 Gene_042 Gene_023\nCell type:",
        "What cell type does this gene expression pattern represent?\nGenes: Gene_010 Gene_005 Gene_030\nCell type:",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n Test {i}:")
        print(f"Prompt: {prompt}")
        try:
            # Need to convert gene names to ranks for the prompt if the LLM was trained on ranks.
            # However, for demo, we'll use the original logic (assuming LLM is smart enough or trained on gene names)
            generated = trainer.generate_text(prompt, max_new_tokens=10)
            print(f"Generated: {generated}")
        except Exception as e:
            print(f" Generation failed: {e}")

    print("\nDemo completed successfully!")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Cell2Sentence Training Demo")
    parser.add_argument("--model-path", type=str, help="Path to local GPT-2 model directory")
    parser.add_argument("--tokenizer-path", type=str, help="Path to local GPT-2 tokenizer directory")
    parser.add_argument("--train", action="store_true", help="Actually train the model (takes time!)")
    parser.add_argument("--data-file", type=str, help="Path to AnnData file (.h5ad)")

    # Check if running in an interactive environment (like Colab)
    if '__file__' not in globals():
        sys.argv = [sys.argv[0]]

    args = parser.parse_args()

    # Use default paths if not provided
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path

    if not model_path and not tokenizer_path:
        print("ℹ No local model paths provided, will use online GPT-2")
    else:
        if model_path and not os.path.exists(model_path):
            print(f"Model path does not exist: {model_path}")
            return
        if tokenizer_path and not os.path.exists(tokenizer_path):
            print(f"Tokenizer path does not exist: {tokenizer_path}")
            return

    try:
        run_demo(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            train_model=args.train,
            data_file=args.data_file
        )
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        # If running in an interactive environment, raise to show the full traceback
        if '__file__' not in globals():
             raise

if __name__ == "__main__":
    main()