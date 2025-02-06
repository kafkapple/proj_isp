import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

class EmotionDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, config: DictConfig):
        self.metadata = metadata
        self.config = config
        self.sample_rate = config.dataset.audio.sample_rate
        self.duration = config.dataset.audio.duration
        self.max_length = config.dataset.audio.max_length
        self.split = config.dataset.split
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load and process audio
        audio = self._load_audio(row['file_path'])
        
        # Get label
        label = self.config.dataset.class_names.index(row['emotion'])
        
        return {
            'audio': audio,
            'label': label,
            'path': row['file_path']
        }
        
    def _load_audio(self, file_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        if len(audio) > self.max_length:
            if self.split == 'train':
                # Random crop for training
                start = np.random.randint(0, len(audio) - self.max_length)
            else:
                # Center crop for validation/test
                start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        else:
            # Pad if shorter
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Convert to tensor
        audio = torch.from_numpy(audio).float()
        
        return audio 