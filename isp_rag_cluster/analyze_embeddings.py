import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from src.data.data_manager import DataManager
from src.models.retriever import Retriever
from src.analysis.embedding_analyzer import EmbeddingAnalyzer
import numpy as np
from pathlib import Path

# .env 파일에서 환경 변수 로드
load_dotenv()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # 데이터 로드
    print("Loading data for embedding analysis...")
    data_manager = DataManager(cfg)
    train_df, _ = data_manager.split_data()
    
    # Retriever 초기화 및 임베딩 생성/로드
    print("\nInitializing retriever...")
    retriever = Retriever(train_df, cfg)
    
    # vector store 경로 확인
    vector_store_path = Path("vector_store")
    index_path = vector_store_path / f"index_{cfg.model.provider}_{cfg.data.n_samples}"
    
    if index_path.exists():
        print(f"\nLoading existing vector store from {index_path}")
        retriever.load_vector_store(index_path)
    else:
        print("\nCreating new vector store...")
        retriever.create_vector_store()
        print(f"\nSaving vector store to {index_path}")
        retriever.save_vector_store(index_path)
    
    # 임베딩 추출
    print("\nExtracting embeddings...")
    embeddings_array = retriever.vector_store.index.reconstruct_n(
        0, retriever.vector_store.index.ntotal
    )
    
    # 레이블 추출
    labels = [
        doc.metadata["emotion"] 
        for doc in retriever.vector_store.docstore._dict.values()
    ]
    
    print(f"Extracted {len(embeddings_array)} embeddings with {embeddings_array.shape[1]} dimensions")
    print(f"Number of labels: {len(labels)}")
    
    # 임베딩 분석
    print("\nAnalyzing embeddings...")
    analyzer = EmbeddingAnalyzer(cfg)
    results = analyzer.analyze(embeddings_array, labels)
    
    # 결과 출력
    print("\nAnalysis Results:")
    print(f"Best number of clusters: {results['best_k']}")
    print(f"Best score ({results['best_k_criterion']['type']}/{results['best_k_criterion']['metric']}): {results['best_score']:.4f}")
    print(f"\nResults saved in: {analyzer.results_dir}")

if __name__ == "__main__":
    main() 