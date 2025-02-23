from typing import Dict, Any
from .base import BaseEmbedding
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingManager(BaseEmbedding):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.embedding_model = self._initialize_embedding()
        
    def _initialize_embedding(self):
        if self.cfg.model.provider == "openai":
            return OpenAIEmbeddings(
                model=self.cfg.model.openai.embedding_model
            )
        else:
            try:
                return HuggingFaceEmbeddings(
                    model_name=self.cfg.model.embeddings.default_model
                )
            except Exception as e:
                print(f"Failed to load default model: {e}")
                return HuggingFaceEmbeddings(
                    model_name=self.cfg.model.embeddings.fallback_model
                )
    
    def embed_query(self, text: str):
        return self.embedding_model.embed_query(text)

    @property
    def embedding_info(self) -> Dict[str, Any]:
        return {
            "provider": self.cfg.model.provider,
            "model": self.embedding_model.__class__.__name__
        }

    def initialize(self):
        """이미 초기화되어 있으므로 패스"""
        pass 