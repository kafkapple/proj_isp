from typing import Dict, List
from omegaconf import DictConfig
from .base_metrics import BaseEmotionMetrics
import logging
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score

class ImageEmotionMetrics(BaseEmotionMetrics):
    def __init__(self, num_classes: int, class_names: List[str], config: DictConfig):
        super().__init__(num_classes, class_names, config)

    def get_classification_report(self) -> str:
        """현재 에포크의 classification report 반환"""
        if not self.all_labels or not self.all_preds:
            return "No predictions available"
        
        y_true = torch.tensor(self.all_labels).cpu().numpy()
        y_pred = torch.tensor(self.all_preds).cpu().numpy()
        
        return classification_report(
            y_true, 
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )

    def compute(self, prefix: str = "") -> Dict[str, torch.Tensor]:
        """모든 메트릭 계산"""
        if not self.all_labels or not self.all_preds:
            return {}
        
        y_true = torch.tensor(self.all_labels)
        y_pred = torch.tensor(self.all_preds)
        
        # 기본 메트릭 계산
        phase = prefix.rstrip('_/')  # train_, val_, test_ 제거
        metrics = {
            f"{phase}/accuracy": accuracy_score(y_true.cpu(), y_pred.cpu()),
            f"{phase}/macro_f1": f1_score(y_true.cpu(), y_pred.cpu(), average='macro'),
            f"{phase}/weighted_f1": f1_score(y_true.cpu(), y_pred.cpu(), average='weighted')
        }
        
        # 클래스별 F1 점수
        class_f1 = f1_score(y_true.cpu(), y_pred.cpu(), average=None)
        for i, score in enumerate(class_f1):
            metrics[f"{phase}/f1_{self.class_names[i]}"] = score
        
        return metrics