import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from typing import Dict, List

class MetricsManager:
    def __init__(self, cfg):
        self.cfg = cfg
        # Use hydra's working directory
        self.results_dir = Path(HydraConfig.get().runtime.output_dir)
        
        # Analysis results save directory
        self.analysis_dir = self.results_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python basic types"""
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

    def save_config(self, config: Dict):
        """Save configuration"""
        # Convert numpy types to Python basic types before saving
        config = self._convert_to_serializable(config)
        with open(self.analysis_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def save_metrics(self, metrics: dict):
        """Save metrics"""
        metrics = self._convert_to_serializable(metrics)
        with open(self.analysis_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def save_predictions(self, predictions: List[Dict]):
        """Save predictions"""
        predictions = self._convert_to_serializable(predictions)
        with open(self.analysis_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

    def save_classification_report(self, y_true: list, y_pred: list, labels: list):
        report = classification_report(y_true, y_pred, labels=labels)
        with open(self.analysis_dir / "classification_report.txt", "w") as f:
            f.write(report)

    def save_confusion_matrix(self, true_labels, pred_labels, class_names):
        """Save confusion matrix visualization"""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot raw counts
        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )
        disp.plot(
            ax=ax1,
            cmap=self.cfg.analysis.embedding.visualization.heatmap.cmap,
            values_format='d'
        )
        ax1.set_title('Confusion Matrix (Counts)')
        
        # Plot normalized values
        cm_normalized = confusion_matrix(
            true_labels, pred_labels, normalize='true'
        )
        disp_norm = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized,
            display_labels=class_names
        )
        disp_norm.plot(
            ax=ax2,
            cmap=self.cfg.analysis.embedding.visualization.heatmap.cmap,
            values_format=self.cfg.analysis.embedding.visualization.heatmap.fmt
        )
        ax2.set_title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "confusion_matrix.png")
        plt.close()

        # Save raw confusion matrix data
        with open(self.analysis_dir / "confusion_matrix.txt", "w", encoding="utf-8") as f:
            np.savetxt(f, cm, fmt="%d") 