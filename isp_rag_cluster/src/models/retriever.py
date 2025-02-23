from pathlib import Path
from .base import BaseRetriever
from .embeddings import EmbeddingManager
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import numpy as np
from typing import List, Dict

class Retriever(BaseRetriever):
    def __init__(self, train_df, cfg):
        super().__init__(cfg)
        self.train_df = train_df
        self.embeddings = EmbeddingManager(cfg)
        self.initialize()
        
    def initialize(self):
        """Vector store 초기화"""
        self.vector_store = self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        index_path = Path("vector_store")
        index_path.mkdir(exist_ok=True)
        
        print("\nCreating documents...")
        documents = [
            Document(
                page_content=str(row[self.cfg.data.column_mapping.text]),
                metadata={"emotion": str(row[self.cfg.data.column_mapping.emotion])}
            ) for _, row in tqdm(self.train_df.iterrows(), desc="Processing documents")
        ]
        
        print(f"\nIndexing {len(documents)} documents...")
        return FAISS.from_documents(
            documents, 
            self.embeddings.embedding_model
        )
    
    def retrieve(self, text: str) -> List[Dict]:
        """유사한 문서 검색
        
        FAISS 검색 파라미터:
        - k: 최종 반환할 문서 수
        - fetch_k: 초기 후보 검색 수 (k보다 커야 함)
        - score_threshold: 유사도 임계값 (L2 거리 기준)
        - search_type: 
            - "similarity": 순수 유사도 기반 검색
            - "mmr": Maximum Marginal Relevance (다양성 고려)
        - nprobe: 검색할 클러스터 수 (높을수록 정확도는 높아지나 속도는 느려짐)
        - ef_search: HNSW 그래프 탐색 범위 (높을수록 정확도는 높아지나 속도는 느려짐)
        """
        if self.cfg.get('debug', {}).get('show_retrieval', False):
            print("\nRetrieving similar documents...")
        
        search_kwargs = {
            'k': self.cfg.model.retriever.fetch_k,
            'search_type': self.cfg.model.retriever.get('search_type', 'similarity'),
            'nprobe': self.cfg.model.retriever.get('nprobe', 10),
            'ef_search': self.cfg.model.retriever.get('ef_search', 40)
        }
        
        # MMR 사용 시 추가 파라미터
        if search_kwargs['search_type'] == 'mmr':
            search_kwargs['lambda_mult'] = self.cfg.model.retriever.mmr.get('lambda_mult', 0.5)
        
        # 후보 문서 검색
        docs = self.vector_store.similarity_search_with_score(
            text,
            **search_kwargs
        )
        
        # 스코어 기반 필터링 (L2 거리가 클수록 유사도가 낮음)
        filtered_docs = [
            (doc, score) for doc, score in docs 
            if score <= (1.0 / self.cfg.model.retriever.score_threshold)
        ]
        
        # k개 문서 선택
        selected_docs = filtered_docs[:self.cfg.model.retriever.k]
        
        # 결과가 없는 경우 처리
        if not selected_docs:
            print("\nWarning: No documents passed the similarity threshold. Using top-k without filtering.")
            selected_docs = docs[:self.cfg.model.retriever.k]
        
        # 결과 로깅
        if self.cfg.get('debug', {}).get('show_retrieval', False):
            print(f"\nRetrieved {len(selected_docs)} similar documents:\n")
            for i, (doc, score) in enumerate(selected_docs, 1):
                print(f"Document {i}:")
                print(f"Content: {doc.page_content}")
                print(f"Emotion: {doc.metadata['emotion']}")
                print(f"Similarity Score: {score:.4f}\n")
        
        # 필요한 형식으로 변환
        return [{
            'page_content': doc.page_content,
            'metadata': doc.metadata,
            'score': float(score)  # 점수도 포함
        } for doc, score in selected_docs]

    def analyze_embeddings(self):
        """임베딩 분석 수행"""
        from src.analysis.embedding_analyzer import EmbeddingAnalyzer
        
        # 문서 임베딩 추출
        embeddings = self.vector_store.docstore._dict.values()
        embeddings_array = np.array([doc.embedding for doc in embeddings])
        
        # 레이블 추출
        labels = [doc.metadata["emotion"] for doc in embeddings]
        
        # 분석 수행
        analyzer = EmbeddingAnalyzer(self.cfg)
        results = analyzer.analyze(embeddings_array, labels)
        
        return results 

    def save_vector_store(self, path: Path):
        """Vector store 저장"""
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(
            folder_path=str(path),
            index_name="index"
        )

    def load_vector_store(self, path: Path):
        """Vector store 로드"""
        self.vector_store = FAISS.load_local(
            folder_path=str(path),
            embeddings=self.embeddings.embedding_model,
            index_name="index",
            allow_dangerous_deserialization=True  # 신뢰할 수 있는 로컬 파일이므로 허용
        )

    def create_vector_store(self):
        """Vector store 생성"""
        print("\nCreating documents...")
        documents = [
            Document(
                page_content=str(row[self.cfg.data.column_mapping.text]),
                metadata={"emotion": str(row[self.cfg.data.column_mapping.emotion])}
            ) for _, row in tqdm(self.train_df.iterrows(), desc="Processing documents")
        ]
        
        print(f"\nIndexing {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(
            documents, 
            self.embeddings.embedding_model
        ) 