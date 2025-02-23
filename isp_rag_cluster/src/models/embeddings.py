from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

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
        self.initialize_embeddings()
        
    def initialize_embeddings(self):
        """Initialize embedding model based on provider"""
        if self.provider == "openai":
            self.embedding_model = OpenAIEmbeddings(
                model=self.cfg.model.openai.embedding_model
            )
            self.embedding_info = {
                "provider": "openai",
                "model": self.cfg.model.openai.embedding_model
            }
        elif self.provider == "lmstudio":
            # Import LMStudioEmbeddings from the root package
            from rag import LMStudioEmbeddings
            self.embedding_model = LMStudioEmbeddings(
                base_url=self.cfg.model.lmstudio.base_url,
                api_key=self.cfg.model.lmstudio.api_key
            )
            # LMStudio는 /embeddings/models 엔드포인트를 지원하지 않으므로 config에서 가져옴
            self.embedding_info = {
                "provider": "lmstudio",
                "model": self.cfg.model.lmstudio.embedding_model,
                "base_url": self.cfg.model.lmstudio.base_url
            }
        else:
            try:
                # Try loading default model
                self.embedding_model = SentenceTransformer(self.cfg.model.embeddings.default_model)
                self.embedding_info = {
                    "provider": "huggingface",
                    "model": self.cfg.model.embeddings.default_model
                }
            except Exception as e:
                logger.warning(f"Failed to load default model: {e}")
                logger.info(f"Falling back to {self.cfg.model.embeddings.fallback_model}")
                # Fallback to lighter model
                self.embedding_model = SentenceTransformer(self.cfg.model.embeddings.fallback_model)
                self.embedding_info = {
                    "provider": "huggingface",
                    "model": self.cfg.model.embeddings.fallback_model
                }

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return current embedding model information"""
        return self.embedding_info

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