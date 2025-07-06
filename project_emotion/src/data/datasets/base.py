from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from pathlib import Path
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base class for all emotion datasets"""
    
    def __init__(self, config: Dict[str, Any], split: str):
        """Initialize dataset"""
        self.config = config
        self.split = split
        self.root_dir = Path(config.dataset.root_dir)
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        pass
        
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Return a sample and its label"""
        pass 