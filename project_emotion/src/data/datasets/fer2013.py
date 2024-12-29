# src/data/fer2013.py
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import torchvision.transforms as transforms
import logging
from typing import Tuple, Optional
import kaggle
from dotenv import load_dotenv

class FER2013Dataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.dataset_path = Path(config.dataset.root_dir)
        
        # 데이터셋 다운로드 확인 및 처리
        self._download_dataset()
        
        # 이미지 경로와 레이블 수집
        self.samples = []
        self._collect_samples()
        
        # 변환 설정
        self.transform = self._get_transforms()
        
        if config.debug.enabled:
            logging.info(f"Loaded {split} dataset with {len(self.samples)} samples")
    
    def _download_dataset(self):
        """FER2013 데이터셋 다운로드"""
        if (self.dataset_path / "train").exists() and (self.dataset_path / "test").exists():
            if self.config.debug.enabled:
                logging.info(f"FER2013 dataset already exists at {self.dataset_path}")
            return
        
        try:
            logging.info("Downloading FER2013 dataset...")
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Kaggle 인증 설정
            load_dotenv()
            os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
            
            if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
                raise ValueError("Kaggle credentials not found in .env file")
            
            # 데이터셋 다운로드
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'msambare/fer2013',
                path=str(self.dataset_path),
                unzip=True
            )
            
            if not (self.dataset_path / "train").exists():
                raise FileNotFoundError(f"Dataset download failed: train directory not found")
                
            logging.info(f"Dataset downloaded successfully to {self.dataset_path}")
            
        except Exception as e:
            logging.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def _collect_samples(self):
        """이미지 경로와 레이블 수집"""
        if self.split == 'test':
            # 테스트셋은 test 폴더 사용
            base_path = self.dataset_path / "test"
        else:
            # train과 val은 train 폴더에서 분할
            base_path = self.dataset_path / "train"
        
        # 각 감정 클래스 폴더 순회
        for emotion_idx, emotion_name in enumerate(self.config.dataset.class_names):
            emotion_path = base_path / emotion_name
            if not emotion_path.exists():
                continue
                
            # 해당 감정의 모든 이미지 파일 수집
            for img_path in emotion_path.glob("*.jpg"):  # 또는 *.png 등 실제 확장자에 맞게 수정
                self.samples.append((str(img_path), emotion_idx))
        
        # train/val 분할 처리
        if self.split != 'test':
            # 전체 데이터 개수
            total_size = len(self.samples)
            # train:val = 0.8:0.2로 분할
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            
            # 랜덤 분할
            train_indices, val_indices = random_split(
                range(total_size), 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드 설정
            )
            
            # 해당하는 split의 샘플만 유지
            if self.split == 'train':
                self.samples = [self.samples[i] for i in train_indices]
            else:  # val
                self.samples = [self.samples[i] for i in val_indices]
    
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