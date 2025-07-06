import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import wandb
import logging
import torch
from typing import Dict, Any, Optional
import torch.nn as nn

class BaseTrainer(pl.LightningModule):
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(config)
        
        # 메트릭 누적을 위한 리스트
        self.training_step_losses = []
        self.validation_step_losses = []
        self.learning_rates = []
        self.current_epoch_steps = 0
        
        # Loss function 초기화
        self.criterion = self._init_criterion()
    
    def _init_criterion(self):
        """Loss function 초기화"""
        if self.config.train.loss.name == "focal":
            from src.focal_loss import FocalLoss
            weights = self._get_class_weights()
            return FocalLoss(
                alpha=weights,
                gamma=self.config.train.loss.focal.gamma  # gamma=2.0
            )
        else:
            weights = self._get_class_weights()
            return torch.nn.CrossEntropyLoss(weight=weights)
            
    def _get_class_weights(self):
        """클래스 가중치 계산"""
        if not self.config.train.loss.use_class_weights:
            return None
            
        return torch.tensor([
            self.config.train.loss.class_weights.manual_weights[name]
            for name in self.config.dataset.class_names
        ])
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.criterion(outputs, batch["label"])
        self._log_step_metrics(loss, "train")
        return loss
        
    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.criterion(outputs, batch["label"])
        self._log_step_metrics(loss, "validation")
        return loss
        
    def test_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.criterion(outputs, batch["label"])
        self.log("test/loss", loss)
        return loss
    
    def _log_step_metrics(self, loss: torch.Tensor, stage: str):
        """스텝별 메트릭 로깅"""
        if stage == "train":
            self.training_step_losses.append(loss.item())
            self.learning_rates.append(self.optimizers().param_groups[0]['lr'])
            self.current_epoch_steps += 1
            
            # 매 N 스텝마다 현재까지의 평균 기록
            step_interval = self.config.logging.step_interval  # 100
            if self.current_epoch_steps % step_interval == 0:
                self._log_interval_metrics()
        else:
            self.validation_step_losses.append(loss.item())
            
        self.log(f"{stage}/loss", loss, prog_bar=True)
    
    def _log_interval_metrics(self):
        """구간별 메트릭 로깅"""
        step_interval = self.config.logging.step_interval
        metrics = self.config.logging.metrics
        
        if 'loss' in metrics:
            # step_interval 동안의 평균 loss 계산
            avg_loss = np.mean(self.training_step_losses[-step_interval:])
            self.log('train/step_loss', avg_loss, prog_bar=True)
            
        if 'learning_rate' in metrics:
            # step_interval 동안의 평균 learning rate 계산
            avg_lr = np.mean(self.learning_rates[-step_interval:])
            self.log('train/learning_rate', avg_lr)

    def on_train_epoch_start(self):
        current_epoch = self.current_epoch + 1
        max_epochs = self.trainer.max_epochs
        logging.info(f"\nStarting Epoch {current_epoch}/{max_epochs}")
    
    def on_validation_epoch_start(self):
        current_epoch = self.current_epoch + 1
        logging.info(f"\nStarting Validation Epoch {current_epoch}")
    
    def configure_optimizers(self):
        # 파라미터 그룹을 3개로 분리
        frozen_params = []
        unfrozen_params = []
        classifier_params = []
        
        # Wav2Vec 파라미터 분류
        for name, param in self.model.named_parameters():
            if 'model' in name:  # wav2vec 모델
                if param.requires_grad:  # unfrozen layers
                    unfrozen_params.append(param)
                else:  # frozen layers
                    frozen_params.append(param)
            else:  # classifier
                classifier_params.append(param)
        
        # 각 그룹별 learning rate 설정
        optimizer = torch.optim.AdamW([
            {'params': frozen_params, 'lr': self.config.train.learning_rate * 0.01},  # 가장 작은 lr
            {'params': unfrozen_params, 'lr': self.config.train.learning_rate * 0.1},  # 중간 lr
            {'params': classifier_params, 'lr': self.config.train.learning_rate}  # 가장 큰 lr
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