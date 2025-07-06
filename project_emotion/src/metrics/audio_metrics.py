from typing import Dict
from .base_metrics import BaseEmotionMetrics
import logging

class AudioEmotionMetrics(BaseEmotionMetrics):
    def compute(self, prefix: str = "", log_wandb: bool = True) -> Dict[str, float]:
        # AudioEmotionMetrics만의 특별한 처리가 필요한 경우 여기에 구현
        return super().compute(prefix, log_wandb) 