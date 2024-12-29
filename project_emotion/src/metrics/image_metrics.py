from typing import Dict
from .base_metrics import BaseEmotionMetrics
import logging

class ImageEmotionMetrics(BaseEmotionMetrics):
    def compute(self, prefix: str = "", log_wandb: bool = True) -> Dict[str, float]:
        phase = prefix.replace('_', '') if prefix else 'eval'
        
        try:
            # 기본 메트릭 계산
            metrics = self.compute_metrics()
            
            # 로깅
            self.log_classification_report(phase)
            
            if log_wandb:
                self.log_wandb_metrics(prefix, phase)
            
            return metrics
        except Exception as e:
            logging.error(f"Error in computing metrics: {e}")
            return {} 