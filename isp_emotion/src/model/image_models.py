from typing import Dict, Any
import torch.nn as nn
import logging
import torch
from torchvision import models

class PretrainedImageModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 모델 초기화
        self.model = self._init_model()
        
        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
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
        
        if config.debug.enabled:
            self._log_model_info()
    
    def _init_model(self) -> nn.Module:
        """모델 초기화"""
        if self.config.model.name == "resnet":
            model = getattr(models, self.config.model.architecture)(
                pretrained=self.config.model.pretrained
            )
            if self.config.model.grayscale:
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif self.config.model.name == "efficientnet":
            model = getattr(models, self.config.model.architecture)(
                pretrained=self.config.model.pretrained
            )
            if self.config.model.grayscale:
                model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.feature_dim = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            
        return model
    
    def forward(self, batch):
        x = batch["image"]
        
        if self.config.debug.enabled:
            logging.debug(f"Input shape: {x.shape}")
        
        features = self.model(x)
        if self.config.debug.enabled:
            logging.debug(f"Backbone output shape: {features.shape}")
        
        features = self.feature_extractor(features)
        logits = self.classifier_head(features)
        
        if self.config.debug.enabled:
            logging.debug(f"Final output shape: {logits.shape}")
        
        return logits
    
    def _log_model_info(self):
        """모델 정보 로깅"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info("\nModel Architecture:")
        logging.info("==================================================")
        logging.info(f"Backbone: {self.config.model.architecture}")
        logging.info(f"Feature dim: {self.feature_dim}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info("==================================================\n") 