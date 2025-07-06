from pathlib import Path
import torch
from typing import Dict, Any, List
import numpy as np

from src.data.datasets.base import BaseDataset
from src.data.utils.audio import AudioProcessor
from src.data.utils.processor import DataProcessor
from src.data.utils.downloader import DataDownloader

class RavdessDataset(BaseDataset):
    """RAVDESS 데이터셋 클래스"""
    
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        super().__init__(config, split)
        self.audio_processor = AudioProcessor()
        self.data_processor = DataProcessor(config)
        self.downloader = DataDownloader(config)
        self._prepare_dataset()
        
    def _init_config(self):
        """설정 초기화"""
        self.root_dir = Path(self.config.dataset.root_dir)
        self.sample_rate = self.config.dataset.audio.sample_rate
        self.duration = self.config.dataset.audio.duration
        self.max_length = self.config.dataset.audio.max_length
        self.normalize = self.config.dataset.audio.normalize
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터셋 아이템 반환"""
        sample = self.samples.iloc[idx]
        
        # 오디오 로드 및 처리
        audio_data, _ = self.audio_processor.load_audio(
            str(self.root_dir / sample['file_path']),
            self.sample_rate,
            self.duration,
            self.normalize
        )
        
        # 패딩/자르기
        audio_data = self.audio_processor.pad_or_truncate(
            audio_data,
            self.max_length,
            mode=self.config.dataset.audio.padding
        )
        
        # 오디오 데이터를 고정된 크기로 만들기
        if len(audio_data) > self.max_length:
            audio_data = audio_data[:self.max_length]
        elif len(audio_data) < self.max_length:
            # 부족한 부분을 0으로 패딩
            padding = np.zeros(self.max_length - len(audio_data))
            audio_data = np.concatenate([audio_data, padding])
        
        # 텐서로 변환 및 shape 확인
        audio_tensor = torch.FloatTensor(audio_data)
        label_tensor = torch.LongTensor([sample['label']])
        
        return {
            'audio': audio_tensor,  # shape: (max_length,)
            'label': label_tensor[0]  # shape: scalar
        }
        
    def _prepare_dataset(self):
        """데이터셋 준비"""
        # 데이터 다운로드
        self.downloader.download_dataset("ravdess")
        
        # 데이터 로드 및 처리
        df = self.data_processor.load_metadata(self.root_dir)
        df = self.data_processor.process_dataset(df, self.split)
        
        # 디버그 모드인 경우 서브셋 추출
        self.samples = self._subset_for_debug(df) 

    def _balance_dataset(self, indices: List[int]) -> List[int]:
        """데이터셋 밸런싱"""
        if not self.config.balance.enabled:
            return indices
        
        # training set에만 적용
        if self.split != "train":
            return indices
        
        if self.config.balance.method == "oversample":
            return self._oversample(indices)
        elif self.config.balance.method == "undersample":
            return self._undersample(indices)
        else:
            raise ValueError(f"Unknown balancing method: {self.config.balance.method}") 