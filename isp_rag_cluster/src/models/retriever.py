from pathlib import Path
from .base import BaseRetriever
from .embeddings import EmbeddingManager, LMStudioEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from src.utils.lmstudio_utils import get_lmstudio_model_info
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Retriever(BaseRetriever):
    def __init__(self, train_df, cfg):
        self.train_df = train_df
        self.cfg = cfg
        self.text_column = cfg.data.column_mapping.text
        self.emotion_column = cfg.data.column_mapping.emotion
        self.provider = cfg.model.provider
        
        # Initialize embeddings
        self.initialize_embeddings()
        
        # Create documents
        print("\nCreating documents...")
        self.documents = self._create_documents()
        print(f"Created {len(self.documents)} documents")
        
    def _create_documents(self):
        """Create documents from dataframe"""
        return [
            Document(
                page_content=row[self.text_column],
                metadata={"emotion": row[self.emotion_column]}
            ) for _, row in self.train_df.iterrows()
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
        
    def retrieve(self, text: str) -> str:
        """Retrieve similar documents"""
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
            (doc, 1 / (1 + score)) for doc, score in docs 
            if 1 / (1 + score) >= self.cfg.model.retriever.score_threshold
        ]
        
        # Select k documents
        selected_docs = filtered_docs[:self.cfg.model.retriever.k]
        
        # Handle case where no documents pass the similarity threshold
        if not selected_docs:
            print("\nWarning: No documents passed the similarity threshold. Using top-k without filtering.")
            selected_docs = docs[:self.cfg.model.retriever.k]
        
        # Format retrieved documents with emotion names and scores
        formatted_docs = []
        for doc, score in selected_docs:
            # emotion ID를 이름으로 변환 (예: "1" -> "joy")
            emotion_id = doc.metadata['emotion']
            emotion_name = self.cfg.data.datasets[self.cfg.data.name].emotions.classes[int(emotion_id)-1]
            
            # similarity score를 percentage로 변환 (L2 distance이므로 역수 사용)
            similarity = min(100, round((1.0 / score) * 100, 1))
            
            formatted_text = (
                f"Text: {doc.page_content}\n"
                f"Emotion: {emotion_name}\n"
                f"Similarity: {similarity}%\n"
            )
            formatted_docs.append(formatted_text)
        
        # Log results
        if self.cfg.get('debug', {}).get('show_retrieval', False):
            print("\nRetrieved documents:")
            for doc in formatted_docs:
                print("-" * 80)
                print(doc)
        
        return "\n\n".join(formatted_docs)

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

    def create_vector_store(self):
        """Create vector store"""
        try:
            print(f"\nIndexing {len(self.documents)} documents...")
            
            # Provider별 모델 정보 출력 방식 다르게 처리
            if self.provider == "lmstudio":
                print(f"Using embedding model: {self.embeddings.model_info}")
            elif self.provider == "openai":
                print(f"Using OpenAI embedding model: {self.cfg.model.openai.embedding_model}")
            
            print(f"First document content: {self.documents[0].page_content}")
            print(f"First document metadata: {self.documents[0].metadata}")
            
            # Test embedding generation
            test_embedding = self.embeddings.embed_query("test query")
            print(f"Test embedding dimension: {len(test_embedding)}")
            
            self.vector_store = FAISS.from_documents(
                self.documents,
                self.embeddings
            )
            print("Vector store created successfully")
            print(f"Vector store size: {self.vector_store.index.ntotal}")
            return self.vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            import traceback
            traceback.print_exc()
            raise

    def save_vector_store(self, path: Path):
        """Save vector store"""
        try:
            print(f"\nSaving vector store to: {path}")
            path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self.vector_store.save_local(
                folder_path=str(path),
                index_name="index"
            )
            
            # Update existing metadata with additional info
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            
            # Add vector store specific info
            metadata.update({
                "num_documents": len(self.train_df),
                "vector_store_size": self.vector_store.index.ntotal,
                "last_updated": datetime.now().isoformat()
            })
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Vector store saved successfully to: {path}")
            
        except Exception as e:
            print(f"Error saving vector store: {e}")
            import traceback
            traceback.print_exc()
            raise

    def load_vector_store(self, path: Path):
        """Load vector store"""
        try:
            print(f"\nLoading vector store from: {path}")
            index_path = path / "index.faiss"
            
            if not path.exists():
                raise FileNotFoundError(f"Vector store path does not exist: {path}")
            
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index file not found: {index_path}")
            
            print(f"Loading FAISS index from: {index_path}")
            
            self.vector_store = FAISS.load_local(
                folder_path=str(path),
                embeddings=self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            
            print(f"Loaded vector store with {self.vector_store.index.ntotal} vectors")
            
            # Load metadata
            metadata_file = path / "store_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    print("\nVector store metadata:")
                    print(json.dumps(metadata, indent=2))
            
            return self.vector_store
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            import traceback
            traceback.print_exc()
            raise

    def initialize_embeddings(self):
        """Initialize embedding model"""
        if self.provider == "lmstudio":
            model_info = get_lmstudio_model_info(
                self.cfg.model.lmstudio.base_url,
                self.cfg.model.lmstudio.api_key,
                model_type="embedding"
            )
            print(f"\nInitializing LMStudio embeddings with model: {model_info['id']}")
            self.embeddings = LMStudioEmbeddings(
                base_url=self.cfg.model.lmstudio.base_url,
                api_key=self.cfg.model.lmstudio.api_key,
                model_info=model_info
            )
        elif self.provider == "openai":
            print(f"\nInitializing OpenAI embeddings with model: {self.cfg.model.openai.embedding_model}")
            self.embeddings = OpenAIEmbeddings(
                model=self.cfg.model.openai.embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY")  # .env에서 API 키 가져오기
            )
        else:
            # OpenAI나 다른 provider의 경우 처리
            raise NotImplementedError(f"Provider {self.provider} not implemented") 