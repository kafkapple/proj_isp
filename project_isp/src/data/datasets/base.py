from torch.utils.data import Dataset
from typing import Dict, Any
import pandas as pd
import logging

class BaseDataset(Dataset):
    """기본 데이터셋 클래스"""
    
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.config = config
        self.split = split
        self.samples = None
        self._init_config()
        
    def _init_config(self):
        """설정 초기화"""
        raise NotImplementedError
        
    def __len__(self):
        if self.samples is None:
            return 0
        return len(self.samples)
        
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def _prepare_dataset(self):
        """데이터셋 준비"""
        raise NotImplementedError
    
    def _subset_for_debug(self, df):
        """디버그용 서브셋 추출"""
        n_samples = self.config.debug.n_samples
        
        # n_samples가 None이면 전체 데이터셋 반환
        if n_samples is None:
            return df
        
        # n_samples가 데이터셋 크기보다 크면 전체 데이터셋 반환
        if len(df) <= n_samples:
            return df
        
        # n_samples 크기만큼 랜덤 샘플링
        return df.sample(n=n_samples, random_state=self.config.dataset.seed) 