from abc import ABC, abstractmethod
from omegaconf import DictConfig

class BaseModel(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    
    @abstractmethod
    def initialize(self):
        pass

class BaseEmbedding(BaseModel):
    @abstractmethod
    def embed_query(self, text: str):
        pass

class BaseRetriever(BaseModel):
    @abstractmethod
    def retrieve(self, query: str):
        pass

class BaseGenerator(BaseModel):
    @abstractmethod
    def generate(self, context: str, query: str):
        pass 