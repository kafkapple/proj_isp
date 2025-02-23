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

    def save_config(self, config_info: dict):
        config_info = self._convert_to_serializable(config_info)
        with open(self.analysis_dir / "config.json", "w") as f:
            json.dump(config_info, f, indent=2)

    def save_metrics(self, metrics: dict):
        metrics = self._convert_to_serializable(metrics)
        with open(self.analysis_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def save_predictions(self, predictions: list):
        predictions = self._convert_to_serializable(predictions)
        df = pd.DataFrame(predictions)
        df.to_csv(self.analysis_dir / "predictions.csv", index=False)

    def save_classification_report(self, y_true: list, y_pred: list, labels: list):
        report = classification_report(y_true, y_pred, labels=labels)
        with open(self.analysis_dir / "classification_report.txt", "w") as f:
            f.write(report)

    def save_confusion_matrix(self, y_true: list, y_pred: list, labels: list):
        """Save confusion matrix using sklearn's ConfusionMatrixDisplay"""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot raw counts
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )
        disp.plot(
            ax=ax1,
            cmap=self.cfg.analysis.embedding.visualization.heatmap.cmap,
            values_format='d'
        )
        ax1.set_title('Confusion Matrix (Counts)')
        
        # Plot normalized values
        cm_normalized = confusion_matrix(
            y_true, y_pred, labels=labels, normalize='true'
        )
        disp_norm = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized,
            display_labels=labels
        )
        disp_norm.plot(
            ax=ax2,
            cmap=self.cfg.analysis.embedding.visualization.heatmap.cmap,
            values_format=self.cfg.analysis.embedding.visualization.heatmap.fmt
        )
        ax2.set_title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close() 