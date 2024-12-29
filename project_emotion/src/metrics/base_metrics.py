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
        """WandB에 메트릭스 로깅"""
        self._log_confusion_matrix(prefix, phase)
        self._log_roc_pr_curves(prefix, phase)
    
    def _log_confusion_matrix(self, prefix: str, phase: str):
        """Confusion Matrix 생성 및 로깅"""
        cm = confusion_matrix(self.all_labels, self.all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'{phase.capitalize()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        wandb.log({f"{prefix}confusion_matrix": wandb.Image(plt)})
        plt.close()

    def _log_roc_pr_curves(self, prefix: str, phase: str):
        """ROC 및 PR 커브 생성 및 로깅"""
        # 각 클래스별 ROC/PR 커브
        for i in range(self.num_classes):
            # One-vs-Rest 방식으로 이진 레이블 생성
            y_true = (np.array(self.all_labels) == i).astype(int)
            y_score = np.array(self.all_probs)[:, i]
            
            # ROC 커브
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr)
            plt.title(f'{self.class_names[i]} ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            wandb.log({f"{prefix}roc_curve_{self.class_names[i]}": wandb.Image(plt)})
            plt.close()
            
            # PR 커브
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision)
            plt.title(f'{self.class_names[i]} Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            wandb.log({f"{prefix}pr_curve_{self.class_names[i]}": wandb.Image(plt)})
            plt.close() 