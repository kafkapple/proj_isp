import librosa
import numpy as np
from typing import Tuple, Optional, List
from omegaconf import DictConfig

class AudioProcessor:
    """Audio processing and augmentation"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.transforms = self._create_transforms()
    
    @staticmethod
    def load_audio(file_path: str, sample_rate: int = 16000, 
                  duration: Optional[float] = None, 
                  normalize: bool = True) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        audio_data, orig_sr = librosa.load(
            file_path,
            sr=sample_rate,
            duration=duration
        )
        
        if normalize:
            audio_data = librosa.util.normalize(audio_data)
        
        return audio_data, orig_sr
    
    def apply_transforms(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply configured transforms"""
        if not self.config.augmentation.enabled:
            return audio
            
        for transform in self.transforms:
            audio = transform(audio, sample_rate)
        return audio
    
    def _create_transforms(self) -> List:
        """Create audio transforms based on config"""
        transforms = []
        if not self.config.augmentation.enabled:
            return transforms
            
        aug_config = self.config.augmentation.transforms
        
        # Add noise
        if aug_config.noise.enabled:
            transforms.append(
                lambda audio, sr: self._add_noise(audio, aug_config.noise.noise_level)
            )
        
        # Random volume
        if aug_config.volume.enabled:
            transforms.append(
                lambda audio, sr: self._adjust_volume(
                    audio, aug_config.volume.min_gain, aug_config.volume.max_gain
                )
            )
        
        # Pitch shift
        if aug_config.pitch_shift.enabled:
            transforms.append(
                lambda audio, sr: self._pitch_shift(audio, sr, aug_config.pitch_shift.steps)
            )
            
        return transforms
    
    @staticmethod
    def _add_noise(audio: np.ndarray, noise_level: float) -> np.ndarray:
        noise = np.random.randn(len(audio))
        return audio + noise_level * noise
    
    @staticmethod
    def _adjust_volume(audio: np.ndarray, min_gain: float, max_gain: float) -> np.ndarray:
        gain = np.random.uniform(min_gain, max_gain)
        return audio * gain
    
    @staticmethod
    def _pitch_shift(audio: np.ndarray, sample_rate: int, steps: List[int]) -> np.ndarray:
        n_steps = np.random.choice(steps)
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps) 