from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class BaseDataset(ABC):
    """Base class for all emotion datasets"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any], split: str):
        """Initialize dataset"""
        self.config = config
        self.split = split
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        pass
        
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Return a sample and its label"""
        pass 