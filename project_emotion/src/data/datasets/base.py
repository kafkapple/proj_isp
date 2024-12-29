from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseDataset(Dataset, ABC):
    def __init__(self, config: Dict[str, Any], split: str):
        self.config = config
        self.split = split
        
    @abstractmethod
    def __len__(self):
        pass
        
    @abstractmethod
    def __getitem__(self, idx):
        pass 