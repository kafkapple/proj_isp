import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from src.data.data_manager import DataManager
from src.models.retriever import Retriever
from src.analysis.embedding_analyzer import EmbeddingAnalyzer
import numpy as np
from pathlib import Path
from src.utils.path_manager import get_vector_store_path

# Load environment variables from .env file
load_dotenv()

@hydra.main(version_base=None, config_path="config", config_name="config_embedding")
def main(cfg: DictConfig):
    # Load data
    print("Loading data for embedding analysis...")
    data_manager = DataManager(cfg)
    train_df, _ = data_manager.split_data()
    
    # Initialize retriever and create/load embeddings
    print("\nInitializing retriever...")
    retriever = Retriever(train_df, cfg)
    
    # Check vector store path
    index_path = get_vector_store_path(cfg)
    
    if index_path.exists():
        print(f"\nLoading existing vector store from {index_path}")
        retriever.load_vector_store(index_path)
    else:
        print("\nCreating new vector store...")
        retriever.create_vector_store()
        print(f"\nSaving vector store to {index_path}")
        retriever.save_vector_store(index_path)
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings_array = retriever.vector_store.index.reconstruct_n(
        0, retriever.vector_store.index.ntotal
    )
    
    # Extract labels
    labels = [
        doc.metadata["emotion"] 
        for doc in retriever.vector_store.docstore._dict.values()
    ]
    
    print(f"Extracted {len(embeddings_array)} embeddings with {embeddings_array.shape[1]} dimensions")
    print(f"Number of labels: {len(labels)}")
    
    # Analyze embeddings
    print("\nAnalyzing embeddings...")
    analyzer = EmbeddingAnalyzer(cfg)
    results = analyzer.analyze(embeddings_array, labels)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Best number of clusters: {results['best_k']}")
    print(f"Best score ({results['best_k_criterion']['type']}/{results['best_k_criterion']['metric']}): {results['best_score']:.4f}")
    print(f"\nResults saved in: {analyzer.results_dir}")

if __name__ == "__main__":
    main() 