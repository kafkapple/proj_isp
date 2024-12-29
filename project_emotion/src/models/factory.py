from typing import Dict, Any
import torch.nn as nn
from omegaconf import OmegaConf
from .wav2vec import Wav2VecEmotionModel
from .image_models import PretrainedImageModel
from ..utils.model_manager import ModelManager
from ..metrics.base_metrics import BaseEmotionMetrics
from ..metrics.image_metrics import ImageEmotionMetrics
from ..metrics.audio_metrics import AudioEmotionMetrics

class ModelFactory:
    @staticmethod
    def create(model_name: str, config: Dict[str, Any]) -> nn.Module:
        # 모델 매니저 초기화
        model_manager = ModelManager(config.model_manager.base_path)
        
        # torchvision 모델은 자동 다운로드되므로 wav2vec만 처리
        if model_name == "wav2vec":
            # wav2vec 모델 다운로드
            model_path = model_manager.download_model(
                model_id=config.model.pretrained,
                model_type="wav2vec2",
                config={"torch_dtype": "float16"}
            )
            # config에 로컬 경로 추가
            config_dict = OmegaConf.to_container(config, resolve=True)
            config_dict['model']['local_path'] = str(model_path)
            config = OmegaConf.create(config_dict)
            return Wav2VecEmotionModel(config)
            
        elif model_name in ["resnet", "efficientnet"]:
            return PretrainedImageModel(config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

def create_metrics(model_name: str, config: Dict) -> BaseEmotionMetrics:
    num_classes = config.dataset.num_classes
    class_names = config.dataset.class_names
    
    if model_name in ["resnet", "efficientnet"]:
        return ImageEmotionMetrics(num_classes, class_names)
    elif model_name == "wav2vec":
        return AudioEmotionMetrics(num_classes, class_names)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
