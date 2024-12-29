from typing import Dict, List
import torch
import numpy as np
import logging
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize

class BaseEmotionMetrics:
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            self.all_preds.extend(preds.cpu().numpy())
            self.all_labels.extend(labels.cpu().numpy())
            self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self, prefix: str = "", log_wandb: bool = True) -> Dict[str, float]:
        """기본 compute 메서드"""
        phase = prefix.replace('_', '') if prefix else 'eval'
        
        try:
            # 기본 메트릭 계산
            report = classification_report(
                self.all_labels, 
                self.all_preds,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            accuracy = accuracy_score(self.all_labels, self.all_preds)
            
            # 메트릭 딕셔너리 구성
            metrics = {
                f"{prefix}accuracy": accuracy,
                f"{prefix}macro_f1": report['macro avg']['f1-score'],
                f"{prefix}weighted_f1": report['weighted avg']['f1-score']
            }
            
            # 각 클래스별 메트릭 추가
            for cls_name in self.class_names:
                metrics.update({
                    f"{prefix}{cls_name}_precision": report[cls_name]['precision'],
                    f"{prefix}{cls_name}_recall": report[cls_name]['recall'],
                    f"{prefix}{cls_name}_f1": report[cls_name]['f1-score']
                })
            
            # 로깅
            self.log_classification_report(phase)
            
            if log_wandb:
                self.log_wandb_metrics(prefix, phase)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in computing metrics: {e}")
            return {}
    
    def log_classification_report(self, phase: str):
        report_str = classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=self.class_names,
            zero_division=0
        )
        log_message = (
            f"\n{'='*50}\n"
            f"{phase.upper()} Classification Report:\n"
            f"{report_str}\n"
            f"{'='*50}\n"
        )
        logging.info(log_message)
    
    def log_wandb_metrics(self, prefix: str, phase: str):
        self._log_confusion_matrix(prefix, phase)
        self._log_roc_pr_curves(prefix, phase)
    
    def _log_confusion_matrix(self, prefix: str, phase: str):
        # confusion matrix 로깅 구현
        ...

    def _log_roc_pr_curves(self, prefix: str, phase: str):
        # ROC/PR curves 로깅 구현
        ... 