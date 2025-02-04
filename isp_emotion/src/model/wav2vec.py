from typing import Dict, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import Wav2Vec2Model
import logging
import numpy as np
from src.metrics.metrics import EmotionMetrics

class Wav2VecModel(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Model components initialization
        self.model = Wav2Vec2Model.from_pretrained(config.model.pretrained)
        
        # Layer freezing setup
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
        
        if config.debug.enabled:
            self._log_model_info()
        
        # Metrics initialization
        self.train_metrics = EmotionMetrics(
            config.dataset.num_classes,
            config.dataset.class_names,
            config
        )
        self.val_metrics = EmotionMetrics(
            config.dataset.num_classes,
            config.dataset.class_names,
            config
        )
        self.test_metrics = EmotionMetrics(
            config.dataset.num_classes,
            config.dataset.class_names,
            config
        )
        
        # Loss function initialization
        self.criterion = self._init_criterion()
        
        # Best metrics tracking
        self.best_metrics = {
            'val/loss': float('inf'),
            'val/accuracy': 0,
            'val/macro_f1': 0,
            'train/accuracy': 0,
            'train/macro_f1': 0,
            'best_epoch': 0
        }
        
    def forward(self, batch):
        x = batch["audio"]
        
        if self.config.debug.enabled:
            logging.debug(f"Input shape: {x.shape}")
        
        outputs = self.model(x).last_hidden_state
        if self.config.debug.enabled:
            logging.debug(f"Wav2Vec output shape: {outputs.shape}")
        
        outputs = outputs.mean(dim=1)
        
        # Evaluation 모드에서는 dropout 비활성화
        if not self.training:
            with torch.no_grad():
                features = self.feature_extractor(outputs)
        else:
            features = self.feature_extractor(outputs)
        
        logits = self.classifier_head(features)
        
        if self.config.debug.enabled:
            logging.debug(f"Final output shape: {logits.shape}")
        
        return logits

    def extract_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Feature extraction method for late fusion"""
        with torch.no_grad():  # no gradient calculation during inference
            return self(batch)
    
    def _log_model_info(self):
        """Model information logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate parameters for each component
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
        """Freeze Wav2Vec layers"""
        # Freeze the entire model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set top N layers to be trainable
        if num_unfrozen_layers > 0:
            for layer in self.model.encoder.layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def training_step(self, batch, batch_idx):
        if self.config.train.mixup.enabled:
            mixed_batch, labels_a, labels_b, lam = self._mixup_batch(batch)
            outputs = self(mixed_batch)
            loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
            
            # Metric 계산을 위한 추가 forward pass
            with torch.no_grad():
                clean_outputs = self(batch)
                self.train_metrics.update(clean_outputs, batch["label"])
        else:
            outputs = self(batch)
            loss = self.criterion(outputs, batch["label"])
            self.train_metrics.update(outputs, batch["label"])
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        self.val_metrics.update(outputs, batch["label"])
        self.log('val/loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        self.train_metrics.set_epoch(self.current_epoch + 1)
        logging.info(f"\nStarting Epoch {self.current_epoch + 1}")

    def on_validation_epoch_start(self):
        self.val_metrics.set_epoch(self.current_epoch + 1)
        logging.info(f"\nStarting Validation Epoch {self.current_epoch + 1}")

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute(prefix="train")
        
        # Log current metrics with explicit batch_size
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True, batch_size=self.config.train.batch_size)
        
        # Store current training metrics for best comparison
        self._current_train_metrics = metrics
        
        # Reset metrics
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute(prefix="val")
        current_loss = metrics.get('val/loss', float('inf'))
        
        # Log current metrics with explicit batch_size
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True, batch_size=self.config.train.batch_size)
        
        # Update best metrics if validation loss improves
        if current_loss < self.best_metrics['val/loss']:
            # Store current metrics as best
            for key in metrics:
                if key.startswith('val/'):
                    self.best_metrics[key] = metrics[key]
            
            # Store current training metrics as best
            if hasattr(self, '_current_train_metrics'):
                for key in self._current_train_metrics:
                    if key.startswith('train/'):
                        self.best_metrics[key] = self._current_train_metrics[key]
            
            self.best_metrics['best_epoch'] = self.current_epoch
            
            # Log best metrics
            for key, value in self.best_metrics.items():
                if key != 'best_epoch':
                    self.log(f"{key}/best", value, sync_dist=True)
            
            # Log best epoch
            self.log("epoch/best", self.best_metrics['best_epoch'], sync_dist=True)
        
        # Reset metrics
        self.val_metrics.reset()
        if hasattr(self, '_current_train_metrics'):
            delattr(self, '_current_train_metrics')  # 임시 저장 데이터 삭제

    def configure_optimizers(self):
        # Parameter groups split
        frozen_params = []
        unfrozen_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'model' in name:
                if param.requires_grad:
                    unfrozen_params.append(param)
                else:
                    frozen_params.append(param)
            else:
                classifier_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': frozen_params, 'lr': self.config.train.learning_rate * 0.01},
            {'params': unfrozen_params, 'lr': self.config.train.learning_rate * 0.1},
            {'params': classifier_params, 'lr': self.config.train.learning_rate}
        ], weight_decay=self.config.train.optimizer.weight_decay)
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.train.scheduler.T_0,
                T_mult=self.config.train.scheduler.T_mult,
                eta_min=self.config.train.scheduler.eta_min
            ),
            "interval": "epoch",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]

    # Helper methods
    def _mixup_batch(self, batch):
        """Mixup augmentation"""
        alpha = self.config.train.mixup.alpha
        lam = np.random.beta(alpha, alpha)
        batch_size = batch["audio"].size(0)
        index = torch.randperm(batch_size)
        
        mixed_audio = lam * batch["audio"] + (1 - lam) * batch["audio"][index]
        return {
            "audio": mixed_audio
        }, batch["label"], batch["label"][index], lam

    def _init_criterion(self):
        """Loss function initialization"""
        if self.config.train.loss.name == "focal":
            from src.losses.focal_loss import FocalLoss
            weights = self._get_class_weights()
            return FocalLoss(
                alpha=weights,
                gamma=self.config.train.loss.focal.gamma
            )
        else:
            weights = self._get_class_weights()
            return torch.nn.CrossEntropyLoss(weight=weights)

    def _get_class_weights(self):
        """Class weight calculation"""
        if not self.config.train.loss.use_class_weights:
            return None
        
        return torch.tensor([
            self.config.train.loss.class_weights.manual_weights[name]
            for name in self.config.dataset.class_names
        ])

    def test_step(self, batch, batch_idx):
        """Test step"""
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        
        # Metrics update
        self.test_metrics = getattr(self, 'test_metrics', EmotionMetrics(
            self.config.dataset.num_classes,
            self.config.dataset.class_names,
            self.config
        ))
        self.test_metrics.update(outputs, batch["label"])
        
        # Log metrics
        self.log('test/loss', loss, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        """Test epoch start"""
        if not hasattr(self, 'test_metrics'):
            self.test_metrics = EmotionMetrics(
                self.config.dataset.num_classes,
                self.config.dataset.class_names,
                self.config
            )
        self.test_metrics.set_epoch(self.current_epoch + 1)
        logging.info(f"\nStarting Test Epoch {self.current_epoch + 1}")

    def on_test_epoch_end(self):
        """Test epoch end"""
        metrics = self.test_metrics.compute(prefix="test")
        self.test_metrics.reset()
        
        for name, value in metrics.items():
            self.log(name, value)

# num_unfrozen_layers: 0  embedding만 사용 (모든 레이어 고정)
# num_unfrozen_layers: N 상위 N개 레이어만 학습
# 예를 들어:
# num_unfrozen_layers: 0 - embedding만 사용
# num_unfrozen_layers: 2 - 상위 2개 레이어만 학습
# num_unfrozen_layers: 12 - 모든 transformer 레이어 학습