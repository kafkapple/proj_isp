from typing import Dict, Any
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import logging
import torch.nn.functional as F
import os
import contextlib
from src.model.base import BaseModel

class Wav2VecModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        # 부모 클래스 초기화 (이제 model info 로깅 안 함)
        super().__init__(config)
        
        # 모든 컴포넌트 초기화
        self.model = Wav2Vec2Model.from_pretrained(config.model.pretrained)
        
        # 레이어 고정 설정
        if config.model.freeze.enabled:
            self._freeze_layers(config.model.freeze.num_unfrozen_layers)
        
        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.train.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.train.dropout)
        )
        
        # Classifier Head
        self.classifier_head = nn.Linear(256, config.dataset.num_classes)
        
        # Gradient Checkpointing 활성화
        self.model.gradient_checkpointing_enable()
        
        # 모든 초기화가 끝난 후 모델 정보 출력
        if config.debug.enabled:
            self._log_model_info()
        
    def forward(self, batch):
        x = batch["audio"]
        
        if self.config.debug.enabled:
            logging.debug(f"Input shape: {x.shape}")
        
        outputs = self.model(x).last_hidden_state
        if self.config.debug.enabled:
            logging.debug(f"Wav2Vec output shape: {outputs.shape}")
        
        outputs = outputs.mean(dim=1)
        features = self.feature_extractor(outputs)
        logits = self.classifier_head(features)
        
        if self.config.debug.enabled:
            logging.debug(f"Final output shape: {logits.shape}")
        
        return logits

    def extract_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Late fusion을 위한 feature extraction 메서드"""
        with torch.no_grad():  # 추론 시에는 gradient 계산 불필요
            return self(batch)
    
    def _log_model_info(self):
        """모델 정보 로깅"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 각 컴포넌트별 파라미터 수 계산
        wav2vec_params = sum(p.numel() for p in self.model.parameters())
        feature_ext_params = sum(p.numel() for p in self.feature_extractor.parameters())
        classifier_params = sum(p.numel() for p in self.classifier_head.parameters())
        
        logging.info("\nModel Architecture:")
        logging.info("==================================================")
        logging.info("Wav2Vec Base Model:")
        logging.info(str(self.model))
        logging.info("\nFeature Extractor:")
        logging.info(str(self.feature_extractor))
        logging.info("\nClassifier Head:")
        logging.info(str(self.classifier_head))
        logging.info("\n==================================================")
        logging.info("Parameter Statistics:")
        logging.info(f"Wav2Vec parameters: {wav2vec_params:,}")
        logging.info(f"Feature Extractor parameters: {feature_ext_params:,}")
        logging.info(f"Classifier Head parameters: {classifier_params:,}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Frozen parameters: {total_params - trainable_params:,}\n")

    def _freeze_layers(self, num_unfrozen_layers: int):
        """Wav2Vec 레이어 고정"""
        # 전체 모델 고정
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 상위 N개 레이어 학습 가능하도록 설정
        if num_unfrozen_layers > 0:
            for layer in self.model.encoder.layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

# num_unfrozen_layers: 0일 때는 embedding만 사용 (모든 레이어 고정)
# num_unfrozen_layers: N일 때는 상위 N개 레이어만 학습
# 예를 들어:
# num_unfrozen_layers: 0 - embedding만 사용
# num_unfrozen_layers: 2 - 상위 2개 레이어만 학습
# num_unfrozen_layers: 12 - 모든 transformer 레이어 학습