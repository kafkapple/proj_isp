import torch.nn as nn
import logging
import torch
from typing import Dict, Any
from src.model.base import BaseModel

class PretrainedImageModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        # 부모 클래스 초기화를 먼저
        super().__init__(config)
        
        # dtype 설정
        self._model_dtype = (
            torch.float32 if config.settings.precision == "32" 
            else torch.float16
        )
        
        # 모델 초기화
        self.model = self._init_model().to(dtype=self._model_dtype)
        self.classifier = self._init_classifier()
        
        # Debug 모드일 때만 모델 정보 출력
        if config.debug.enabled:
            self._log_model_info()
    
    def _init_model(self):
        """모델 초기화"""
        if self.config.model.name == "resnet":
            model = getattr(models, self.config.model.architecture)(
                pretrained=self.config.model.pretrained
            )
            if self.config.model.grayscale:
                # Grayscale 입력을 위한 첫 번째 conv layer 수정
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # FC layer 제거
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif self.config.model.name == "efficientnet":
            model = getattr(models, self.config.model.architecture)(pretrained=self.config.model.pretrained)
            if self.config.model.grayscale:
                # Grayscale 입력을 위한 첫 번째 conv layer 수정
                model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            # Classifier 제거
            self.feature_dim = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            
        return model
    
    def _init_classifier(self):
        """분류기 초기화"""
        layers = []
        in_features = self.feature_dim
        
        # activation 함수 매핑
        activation_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'leakyrelu': nn.LeakyReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        
        # Hidden layers
        for hidden_size in self.config.model.classifier.hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size) if self.config.model.classifier.use_batch_norm else nn.Identity(),
                activation_map[self.config.model.classifier.activation.lower()](),
                nn.Dropout(self.config.model.classifier.dropout)
            ])
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, self.config.dataset.num_classes))
        
        return nn.Sequential(*layers)
    
    def _freeze_layers(self):
        """백본 레이어 고정"""
        for param in self.model.parameters():
            param.requires_grad = False
            
        if self.config.model.name == "resnet":
            # ResNet의 마지막 N개 레이어 학습 가능하도록 설정
            layers_to_unfreeze = self.config.model.freeze.num_unfrozen_layers
            if layers_to_unfreeze > 0:
                for param in self.model.layer4[-layers_to_unfreeze:].parameters():
                    param.requires_grad = True
                    
        elif self.config.model.name == "efficientnet":
            # EfficientNet의 마지막 N개 블록 학습 가능하도록 설정
            blocks_to_unfreeze = self.config.model.freeze.num_unfrozen_layers
            if blocks_to_unfreeze > 0:
                total_blocks = len(self.model.features)
                for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = True
    
    def _log_model_info(self):
        """모델 정보 로깅"""
        if not self.config.debug.model_summary:
            return
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"\nModel Architecture: {self.config.model.architecture}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Frozen parameters: {total_params - trainable_params:,}\n")
    
    def forward(self, batch):
        x = batch["image"]
        features = self.model(x)
        return self.classifier(features)
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        
        # 예측값 업데이트
        preds = torch.argmax(outputs, dim=1)
        self.val_metrics.update(preds, batch["label"])  # outputs -> preds로 변경
        
        self.log("validation/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        
        # 예측값 업데이트
        preds = torch.argmax(outputs, dim=1)
        self.test_metrics.update(preds, batch["label"])  # outputs -> preds로 변경
        
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        # Classification report 출력 (리셋 전에)
        current_epoch = self.current_epoch + 1
        logging.info(f"\nTrain Epoch {current_epoch} Classification Report:")
        logging.info(self.train_metrics.get_classification_report())
        
        # 메트릭 계산 및 로깅
        metrics = self.train_metrics.compute(prefix="train")
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        
        # 메트릭 리셋
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        # Classification report 출력 (리셋 전에)
        current_epoch = self.current_epoch + 1
        logging.info(f"\nValidation Epoch {current_epoch} Classification Report:")
        logging.info(self.val_metrics.get_classification_report())
        
        # 메트릭 계산 및 로깅
        metrics = self.val_metrics.compute(prefix="validation")
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True, sync_dist=True)
        
        # 메트릭 리셋
        self.val_metrics.reset()
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute(prefix="test")
        self.test_metrics.reset()
        
        # 모든 메트릭스 로깅
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        
        # 주요 메트릭 반환
        return {
            "test/macro_f1": metrics["test/macro_f1"],
            "test/weighted_f1": metrics["test/weighted_f1"],
            "test/accuracy": metrics["test/accuracy"]
        }
    
    def on_train_epoch_start(self):
        self.train_metrics.set_epoch(self.current_epoch + 1)
    
    def on_validation_epoch_start(self):
        self.val_metrics.set_epoch(self.current_epoch + 1)
    
    def on_test_epoch_start(self):
        self.test_metrics.set_epoch(self.current_epoch + 1) 