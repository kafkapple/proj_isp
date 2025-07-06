import librosa
import numpy as np
import torch
from typing import Tuple, Optional

class AudioProcessor:
    """오디오 처리를 위한 유틸리티 클래스"""
    
    @staticmethod
    def load_audio(file_path: str, sample_rate: int = 16000, 
                  duration: Optional[float] = None, 
                  normalize: bool = True) -> Tuple[np.ndarray, int]:
        """오디오 파일 로드 및 전처리"""
        audio_data, orig_sr = librosa.load(
            file_path,
            sr=sample_rate,
            duration=duration
        )
        
        if normalize:
            audio_data = librosa.util.normalize(audio_data)
        
        return audio_data, orig_sr
    
    @staticmethod
    def augment_audio(audio: np.ndarray, sample_rate: int,
                     noise_level: float = 0.005,
                     pitch_shift: Optional[int] = None,
                     time_mask_param: Optional[int] = None) -> np.ndarray:
        """오디오 데이터 증강"""
        augmented = audio.copy()
        
        # 노이즈 추가
        if noise_level > 0:
            noise = np.random.randn(len(audio))
            augmented += noise_level * noise
        
        # 피치 시프트
        if pitch_shift is not None:
            augmented = librosa.effects.pitch_shift(
                augmented,
                sr=sample_rate,
                n_steps=pitch_shift
            )
        
        # 시간 마스킹
        if time_mask_param is not None:
            mask_size = np.random.randint(0, time_mask_param)
            mask_start = np.random.randint(0, len(audio) - mask_size)
            augmented[mask_start:mask_start + mask_size] = 0
        
        return augmented
    
    @staticmethod
    def pad_or_truncate(audio: np.ndarray, target_length: int,
                       mode: str = "repeat") -> np.ndarray:
        """오디오 길이 조정"""
        if len(audio) > target_length:
            return audio[:target_length]
        
        padding_length = target_length - len(audio)
        
        if mode == "zero":
            return np.pad(audio, (0, padding_length))
        elif mode == "repeat":
            num_repeats = padding_length // len(audio) + 1
            repeated = np.tile(audio, num_repeats)
            return repeated[:target_length]
        else:
            raise ValueError(f"Unknown padding mode: {mode}") 