from typing import Dict, Any
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from .wav2vec import Wav2VecModel
from .image_models import PretrainedImageModel
from ..metrics.base_metrics import BaseEmotionMetrics
from ..metrics.image_metrics import ImageEmotionMetrics
from ..metrics.audio_metrics import AudioEmotionMetrics

class ModelFactory:
    @staticmethod
    def create(config: DictConfig):
        """모델과 트레이너 생성"""
        model = ModelFactory._create_model(config)
        metrics = ModelFactory.create_metrics(config.model.name, config)
        
        # 순환 import 방지를 위해 import 위치 이동
        from src.trainers.emotion_trainer import EmotionTrainer
        return EmotionTrainer(model, config, metrics)
    
    @staticmethod
    def _create_model(config: DictConfig):
        """순수 모델 생성"""
        if config.model.name == "wav2vec":
            return Wav2VecModel(config)
        elif config.model.name == "efficientnet":
            return PretrainedImageModel(config)
        else:
            raise ValueError(f"Unknown model: {config.model.name}")

    @staticmethod
    def create_metrics(model_name: str, config: DictConfig) -> BaseEmotionMetrics:
        """메트릭스 생성"""
        num_classes = config.dataset.num_classes
        class_names = config.dataset.class_names
        
        if model_name in ["resnet", "efficientnet"]:
            return ImageEmotionMetrics(num_classes, class_names, config)
        elif model_name == "wav2vec":
            return AudioEmotionMetrics(num_classes, class_names, config)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
