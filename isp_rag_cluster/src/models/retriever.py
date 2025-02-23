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
        self.documents = None
        self.vector_store = None
        self._create_documents()  # documents만 생성
        
    def _create_documents(self):
        """Create documents without initializing vector store"""
        if self.documents is None:
            print("\nCreating documents...")
            self.documents = [
                Document(
                    page_content=str(row[self.cfg.data.column_mapping.text]),
                    metadata={"emotion": str(row[self.cfg.data.column_mapping.emotion])}
                ) for _, row in tqdm(self.train_df.iterrows(), desc="Processing documents")
            ]

    def initialize(self):
        """Initialize both documents and vector store"""
        self._create_documents()
        if self.vector_store is None:
            print(f"\nIndexing {len(self.documents)} documents...")
            self.vector_store = FAISS.from_documents(
                self.documents,
                self.embeddings.embedding_model
            )
        
    def retrieve(self, text: str) -> List[Dict]:
        """Retrieve similar documents
        
        FAISS search parameters:
        - k: number of documents to return
        - fetch_k: initial candidate search count (must be greater than k)
        - score_threshold: similarity threshold (L2 distance based)
        - search_type: 
            - "similarity": pure similarity-based search
            - "mmr": Maximum Marginal Relevance (consider diversity)
        - nprobe: number of clusters to search (higher means higher accuracy but slower speed)
        - ef_search: HNSW graph search range (higher means higher accuracy but slower speed)
        """
        if self.cfg.get('debug', {}).get('show_retrieval', False):
            print("\nRetrieving similar documents...")
        
        search_kwargs = {
            'k': self.cfg.model.retriever.fetch_k,
            'search_type': self.cfg.model.retriever.get('search_type', 'similarity'),
            'nprobe': self.cfg.model.retriever.get('nprobe', 10),
            'ef_search': self.cfg.model.retriever.get('ef_search', 40)
        }
        
        # Additional parameters when using MMR
        if search_kwargs['search_type'] == 'mmr':
            search_kwargs['lambda_mult'] = self.cfg.model.retriever.mmr.get('lambda_mult', 0.5)
        
        # Search candidate documents
        docs = self.vector_store.similarity_search_with_score(
            text,
            **search_kwargs
        )
        
        # Score-based filtering (L2 distance means lower similarity)
        filtered_docs = [
            (doc, score) for doc, score in docs 
            if score <= (1.0 / self.cfg.model.retriever.score_threshold)
        ]
        
        # Select k documents
        selected_docs = filtered_docs[:self.cfg.model.retriever.k]
        
        # Handle case where no documents pass the similarity threshold
        if not selected_docs:
            print("\nWarning: No documents passed the similarity threshold. Using top-k without filtering.")
            selected_docs = docs[:self.cfg.model.retriever.k]
        
        # Log results
        if self.cfg.get('debug', {}).get('show_retrieval', False):
            print(f"\nRetrieved {len(selected_docs)} similar documents:\n")
            for i, (doc, score) in enumerate(selected_docs, 1):
                print(f"Document {i}:")
                print(f"Content: {doc.page_content}")
                print(f"Emotion: {doc.metadata['emotion']}")
                print(f"Similarity Score: {score:.4f}\n")
        
        # Convert to required format
        return [{
            'page_content': doc.page_content,
            'metadata': doc.metadata,
            'score': float(score)  # Include score
        } for doc, score in selected_docs]

    def analyze_embeddings(self):
        """Perform embedding analysis"""
        from src.analysis.embedding_analyzer import EmbeddingAnalyzer
        
        # Extract document embeddings
        embeddings = self.vector_store.docstore._dict.values()
        embeddings_array = np.array([doc.embedding for doc in embeddings])
        
        # Extract labels
        labels = [doc.metadata["emotion"] for doc in embeddings]
        
        # Perform analysis
        analyzer = EmbeddingAnalyzer(self.cfg)
        results = analyzer.analyze(embeddings_array, labels)
        
        return results 

    def save_vector_store(self, path: Path):
        """Save vector store"""
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(
            folder_path=str(path),
            index_name="index"
        )

    def load_vector_store(self, path: Path):
        """Load vector store"""
        self.vector_store = FAISS.load_local(
            folder_path=str(path),
            embeddings=self.embeddings.embedding_model,
            index_name="index",
            allow_dangerous_deserialization=True  # Trustworthy local file so allowed
        )

    def create_vector_store(self):
        """Create vector store"""
        # No need to recreate documents as they're already created in initialize()
        print(f"\nIndexing {len(self.documents)} documents...")
        self.vector_store = FAISS.from_documents(
            self.documents, 
            self.embeddings.embedding_model
        ) 