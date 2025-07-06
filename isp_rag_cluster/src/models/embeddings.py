from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.provider = cfg.model.provider
        self.embedding_model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize appropriate embedding model"""
        if self.provider == "openai":
            return OpenAIEmbeddings(
                model=self.cfg.model.openai.embedding_model
            )
        elif self.provider == "lmstudio":
            return LMStudioEmbeddings(
                base_url=self.cfg.model.lmstudio.base_url,
                api_key=self.cfg.model.lmstudio.api_key
            )
        else:
            try:
                # Try loading default model
                return SentenceTransformer(self.cfg.model.embeddings.default_model)
            except Exception as e:
                logger.warning(f"Failed to load default model: {e}")
                logger.info(f"Falling back to {self.cfg.model.embeddings.fallback_model}")
                # Fallback to lighter model
                return SentenceTransformer(self.cfg.model.embeddings.fallback_model)

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return current embedding model information"""
        if self.provider == "lmstudio":
            return self.embedding_model.model_info
        else:
            return {
                "provider": self.provider,
                "model": self.embedding_model.__class__.__name__
            }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if hasattr(self.embedding_model, "embed_documents"):
            return self.embedding_model.embed_documents(texts)
        # For SentenceTransformer
        return self.embedding_model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        if hasattr(self.embedding_model, "embed_query"):
            return self.embedding_model.embed_query(text)
        # For SentenceTransformer
        return self.embedding_model.encode([text])[0].tolist()

    def initialize(self):
        """Already initialized, so pass"""
        pass 

class LMStudioEmbeddings(Embeddings):
    """LMStudio embeddings wrapper for LangChain"""
    
    def __init__(self, base_url: str, api_key: str = "lm-studio", model_info: dict = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_info = model_info or {"id": "unknown"}
        print(f"Initialized with model info: {self._model_info}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            model_id = self._model_info.get("id", "text-embedding-bge-m3")
            print(f"Using model ID for embedding: {model_id}")
            
            response = self.client.embeddings.create(
                model=model_id,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error in embed_documents: {e}")
            print(f"Current model info: {self._model_info}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        try:
            model_id = self._model_info.get("id", "text-embedding-bge-m3")
            print(f"Using model ID for query: {model_id}")
            
            response = self.client.embeddings.create(
                model=model_id,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in embed_query: {e}")
            print(f"Current model info: {self._model_info}")
            raise

    @property
    def model_info(self) -> dict:
        """Return model information"""
        return {
            "provider": "lmstudio",
            "model": self._model_info
        }
    
    @model_info.setter
    def model_info(self, value: dict):
        """Set model information"""
        print(f"Setting model info: {value}")
        self._model_info = value 