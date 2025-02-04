from src.trainers.base import BaseTrainer
from src.metrics.base_metrics import BaseEmotionMetrics
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch

class EmotionTrainer(BaseTrainer):
    def __init__(self, model, config, metrics: BaseEmotionMetrics):
        super().__init__(model, config)
        self.train_metrics = metrics
        self.val_metrics = metrics.__class__(
            config.dataset.num_classes,
            config.dataset.class_names,
            config
        )
        self.test_metrics = metrics.__class__(
            config.dataset.num_classes,
            config.dataset.class_names,
            config
        )
    
    def training_step(self, batch, batch_idx):
        # Mixup 적용
        if self.config.train.mixup.enabled:
            mixed_batch, labels_a, labels_b, lam = self._mixup_batch(batch)
            outputs = self(mixed_batch)
            loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
        else:
            outputs = self(batch)
            loss = self.criterion(outputs, batch["label"])
        
        self.train_metrics.update(outputs, batch["label"])
        self._log_step_metrics(loss, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        self.val_metrics.update(outputs, batch["label"])
        self._log_step_metrics(loss, "val")
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        self.test_metrics.update(outputs, batch["label"])
        self.log("test/loss", loss)
        return loss
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train_metrics.set_epoch(self.current_epoch + 1)
    
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_metrics.set_epoch(self.current_epoch + 1)
    
    def on_test_epoch_start(self):
        self.test_metrics.set_epoch(self.current_epoch + 1)
    
    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")
        
    def on_validation_epoch_end(self):
        self._log_epoch_metrics("validation")
        
    def on_test_epoch_end(self):
        """테스트 에포크 종료 시 메트릭 처리"""
        metrics = self.test_metrics.compute(prefix="test")
        
        # 메트릭 로깅
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
            
        # WandB 시각화
        self.test_metrics.log_metrics_to_wandb("test")
        
        # 메트릭 리셋
        self.test_metrics.reset()
        
        return {
            "test/macro_f1": metrics["test/macro_f1"],
            "test/weighted_f1": metrics["test/weighted_f1"],
            "test/accuracy": metrics["test/accuracy"]
        }
    
    def _log_epoch_metrics(self, stage: str):
        """에포크 단위 메트릭 로깅"""
        mapping = {
            "train": ("train", "train_metrics"),
            "validation": ("val", "val_metrics"),
            "test": ("test", "test_metrics")
        }
        
        prefix, metrics_attr = mapping[stage]
        metrics = getattr(self, metrics_attr)
        computed_metrics = metrics.compute(prefix=prefix)
        current_epoch = self.current_epoch + 1
        
        # 로컬에 저장 (메트릭 리셋 전에)
        output_dir = Path(self.config.dirs.outputs)
        
        # 1. Classification Report 저장
        report_text = metrics.get_classification_report()
        report_path = output_dir / "reports" / f"{stage}_report_epoch_{current_epoch}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text)
        
        # 2. Confusion Matrix 저장
        cm_path = output_dir / "metrics" / f"{stage}_confusion_matrix_epoch_{current_epoch}.png"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        metrics.plot_confusion_matrix()  # confusion matrix 그리기
        plt.savefig(cm_path)
        plt.close()
        
        # 3. PR & ROC Curves 저장
        curves_path = output_dir / "metrics" / f"{stage}_curves_epoch_{current_epoch}.png"
        plt.figure(figsize=(12, 5))
        
        # subplot으로 PR과 ROC curve 함께 저장
        plt.subplot(1, 2, 1)
        metrics.plot_pr_curves()
        plt.title(f"{stage.title()} PR Curves - Epoch {current_epoch}")
        
        plt.subplot(1, 2, 2)
        metrics.plot_roc_curves()
        plt.title(f"{stage.title()} ROC Curves - Epoch {current_epoch}")
        
        plt.tight_layout()
        plt.savefig(curves_path)
        plt.close()
        
        # WandB 로깅
        metrics.log_metrics_to_wandb(prefix)
        
        # Classification Report 콘솔 출력 추가
        logging.info(f"\n{stage.title()} Epoch {current_epoch} Classification Report:")
        logging.info("="*50)
        logging.info(report_text)  # 이미 생성된 report_text 사용
        logging.info("="*50)
        
        # 메트릭 계산 및 로깅
        for name, value in computed_metrics.items():
            self.log(name, value, prog_bar=True)
        
        # 메트릭 리셋 (모든 저장 작업 후에)
        metrics.reset() 

    def configure_optimizers(self):
        # 부모 클래스의 optimizer 설정 사용
        return super().configure_optimizers()

    def _mixup_batch(self, batch):
        """Mixup 데이터 증강"""
        alpha = self.config.train.mixup.alpha
        lam = np.random.beta(alpha, alpha)
        batch_size = batch["audio"].size(0)
        index = torch.randperm(batch_size)
        
        mixed_audio = lam * batch["audio"] + (1 - lam) * batch["audio"][index]
        return {
            "audio": mixed_audio
        }, batch["label"], batch["label"][index], lam 