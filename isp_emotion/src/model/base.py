import torch.nn as nn
import logging
from typing import Dict, Any

class BaseModel(nn.Module):
    """순수 모델 구조를 정의하는 기본 클래스"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
    
    def forward(self, batch):
        """순수 forward pass만 구현"""
        raise NotImplementedError 