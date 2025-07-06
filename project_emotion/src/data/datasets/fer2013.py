# src/data/fer2013.py
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import torchvision.transforms as transforms
import logging
from typing import Tuple, Optional, Dict, Any, List
import kaggle
from dotenv import load_dotenv
from src.data.utils.download import DatasetDownloader
from src.data.datasets.base import BaseDataset
import random

class FER2013Dataset(BaseDataset):
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        super().__init__(config, split)
        
        self.image_size = config.dataset.image.size
        self.transform = self._get_transforms()
        
        # 전체 train 데이터 로드
        if split == "test":
            self.samples = self._collect_samples("test")
        else:
            # train 데이터를 train/val로 분할
            all_train_samples = self._collect_samples("train")
            train_size = int(len(all_train_samples) * config.dataset.splits.ratios.train)
            
            # 랜덤 시드 설정
            random.seed(config.dataset.seed)
            indices = list(range(len(all_train_samples)))
            random.shuffle(indices)
            
            if split == "train":
                split_indices = indices[:train_size]
            else:  # val
                split_indices = indices[train_size:]
                
            self.samples = [all_train_samples[i] for i in split_indices]
    
    def _collect_samples(self, split_name: str) -> List[Tuple[Path, int]]:
        """데이터셋 샘플 수집"""
        samples = []
        split_dir = self.root_dir / split_name
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")
            
        # 각 클래스 디렉토리에서 이미지 수집
        for class_idx, class_name in enumerate(self.config.dataset.class_names):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob("*.jpg"):  # 또는 다른 이미지 확장자
                samples.append((img_path, class_idx))
                
        return samples
    
    def _get_transforms(self):
        """데이터 변환 설정"""
        if self.split == 'train' and self.config.dataset.augmentation.enabled:
            return transforms.Compose([
                transforms.RandomRotation(self.config.dataset.augmentation.rotation_range),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    self.config.dataset.image.size,
                    scale=(0.8, 1.0)
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.config.dataset.image.mean, 
                                  self.config.dataset.image.std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.config.dataset.image.size),
                transforms.ToTensor(),
                transforms.Normalize(self.config.dataset.image.mean, 
                                  self.config.dataset.image.std)
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # grayscale로 로드
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }