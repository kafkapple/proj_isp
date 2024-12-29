from torch.utils.data import DataLoader
from typing import Dict, Any, List
import torch
from ..datasets.base import BaseDataset
import logging

class BaseDataLoader:
    def __init__(self, dataset: BaseDataset, config: Dict[str, Any]):
        self.dataset = dataset
        self.config = config
        
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """배치 데이터 처리"""
        audio = torch.stack([item['audio'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"1. Batch audio shape before processing: {audio.shape}")
        
        # 차원 조정: [batch_size, channels, sequence_length]
        if len(audio.shape) == 4:  # [B, 1, 1, seq_len]
            audio = audio.squeeze(2)  # [B, 1, seq_len]
        
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"2. Batch audio shape after processing: {audio.shape}")
        
        return {
            'audio': audio,
            'label': labels
        }
        
    def get_loader(self, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=self.collate_fn
        ) 

def get_collate_fn(dataset_name: str):
    """데이터셋 타입에 따른 collate_fn 반환"""
    
    def audio_collate_fn(batch):
        """오디오 데이터셋용 collate_fn"""
        audio = torch.stack([item['audio'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'audio': audio, 'label': labels}
    
    def image_collate_fn(batch):
        """이미지 데이터셋용 collate_fn"""
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'image': images, 'label': labels}
    
    # 데이터셋 이름에 따라 적절한 collate_fn 반환
    if dataset_name in ['ravdess']:
        return audio_collate_fn
    elif dataset_name in ['fer2013']:
        return image_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}") 