from typing import Dict, Any
import torch.nn as nn
from omegaconf import DictConfig
from src.model.wav2vec import Wav2VecModel
from .image_models import PretrainedImageModel
from ..metrics.metrics import EmotionMetrics
import logging

class ModelFactory:
    @staticmethod
    def create(config: DictConfig):
        """Create model instance based on config"""
        model_name = config.model.name.lower()
        
        if model_name == "wav2vec":
            model = Wav2VecModel(config)
            logging.info(f"Created model: {type(model)}")  # 모델 타입 로깅
            return model
        elif model_name == "efficientnet":
            return PretrainedImageModel(config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def create_metrics(model_name: str, config: DictConfig) -> EmotionMetrics:
        """메트릭스 생성"""
        num_classes = config.dataset.num_classes
        class_names = config.dataset.class_names
        return EmotionMetrics(num_classes, class_names, config)
