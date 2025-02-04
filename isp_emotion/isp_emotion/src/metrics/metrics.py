from typing import Dict, List
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

class EmotionMetrics:
    def __init__(self, num_classes: int, class_names: List[str], config: DictConfig):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.config = config
        self.current_epoch = 0
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        if isinstance(preds, torch.Tensor):
            if preds.dim() > 1:
                preds = torch.argmax(preds, dim=1)
            preds = preds.cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)
    
    def get_classification_report(self) -> str:
        """return classification report for current epoch"""
        if not self.all_labels or not self.all_preds:
            return "No predictions available"
            
        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0
        )
        
        # print to console
        logging.info("\nClassification Report:")
        logging.info("="*50)
        logging.info(report)
        logging.info("="*50)
        
        return report
    
    def compute(self, prefix: str = "") -> Dict[str, float]:
        """Compute all metrics"""
        if not self.all_labels or not self.all_preds:
            return {}
        
        y_true = torch.tensor(self.all_labels)
        y_pred = torch.tensor(self.all_preds)
        
        # 기본 메트릭스 계산
        metrics = {
            f"{prefix}/accuracy": accuracy_score(y_true.cpu(), y_pred.cpu()),
            f"{prefix}/macro_f1": f1_score(y_true.cpu(), y_pred.cpu(), average='macro'),
            f"{prefix}/weighted_f1": f1_score(y_true.cpu(), y_pred.cpu(), average='weighted')
        }
        
        # 클래스별 F1 점수
        class_f1 = f1_score(y_true.cpu(), y_pred.cpu(), average=None)
        for i, score in enumerate(class_f1):
            metrics[f"{prefix}/f1_{self.class_names[i]}"] = score
        
        # Classification Report 출력 및 로깅
        report = classification_report(
            y_true.cpu(),
            y_pred.cpu(),
            target_names=self.class_names,
            zero_division=0
        )
        
        print(f"\n{prefix.upper()} Classification Report - Epoch {self.current_epoch}:")
        print("="*50)
        print(report)
        print("="*50)
        
        return metrics

    def set_epoch(self, epoch: int):
        """Set current epoch number"""
        self.current_epoch = epoch 