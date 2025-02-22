import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

class MetricsManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _convert_to_serializable(self, obj):
        """numpy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        return obj

    def save_config(self, config_info: dict):
        config_info = self._convert_to_serializable(config_info)
        with open(self.results_dir / "config.json", "w") as f:
            json.dump(config_info, f, indent=2)

    def save_metrics(self, metrics: dict):
        metrics = self._convert_to_serializable(metrics)
        with open(self.results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def save_predictions(self, predictions: list):
        predictions = self._convert_to_serializable(predictions)
        df = pd.DataFrame(predictions)
        df.to_csv(self.results_dir / "predictions.csv", index=False)

    def save_classification_report(self, y_true: list, y_pred: list, labels: list):
        report = classification_report(y_true, y_pred, labels=labels)
        with open(self.results_dir / "classification_report.txt", "w") as f:
            f.write(report)

    def save_confusion_matrix(self, y_true: list, y_pred: list, labels: list):
        """혼동 행렬 저장"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # 정규화
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 두 개의 혼동 행렬 (원본과 정규화) 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 원본 혼동 행렬
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=labels, yticklabels=labels,
                    cmap=self.cfg.analysis.embedding.visualization.heatmap.cmap,
                    ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 정규화된 혼동 행렬
        sns.heatmap(cm_normalized, annot=True, 
                    fmt=self.cfg.analysis.embedding.visualization.heatmap.fmt,
                    xticklabels=labels, yticklabels=labels,
                    cmap=self.cfg.analysis.embedding.visualization.heatmap.cmap,
                    ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "confusion_matrix.png", dpi=300)
        plt.close() 