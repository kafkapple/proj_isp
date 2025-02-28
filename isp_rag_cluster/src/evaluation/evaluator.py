from .metrics import MetricsManager
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from hydra.core.hydra_config import HydraConfig

class Evaluator:
    def __init__(self, analyzer, cfg):
        self.analyzer = analyzer
        self.cfg = cfg
        self.metrics_manager = MetricsManager(cfg)
        self.emotion_classes = cfg.emotions.classes

    def evaluate(self, val_df):
        """Evaluate model performance"""
        print("\nStarting evaluation...")
        
        # Get predictions
        true_emotions = []
        pred_emotions = []
        predictions = []
        
        print(f"\nEvaluating {len(val_df)} samples:")
        for _, row in tqdm(val_df.iterrows(), 
                          desc="Evaluating emotions", 
                          total=len(val_df),
                          unit="samples"):
            text = row[self.cfg.data.column_mapping.text]
            
            # Convert numeric emotion label to string
            emotion_id = str(row[self.cfg.data.column_mapping.emotion])
            true_emotion = self.cfg.emotions.classes[int(emotion_id) - 1]  # 1-based to 0-based indexing
            
            # Get prediction
            pred_emotion = self.analyzer.analyze_emotion(text)
            
            true_emotions.append(true_emotion)
            pred_emotions.append(pred_emotion)
            predictions.append({
                "text": text,
                "true_emotion": true_emotion,
                "pred_emotion": pred_emotion
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_emotions, pred_emotions)
        
        # Save results
        self._save_results(predictions, true_emotions, pred_emotions, metrics)
        
        return metrics

    def _calculate_metrics(self, true_emotions, pred_emotions):
        # Ignore all warning messages
        warnings.filterwarnings('ignore')
        
        # Use only actual labels
        unique_labels = sorted(list(set(true_emotions) | set(pred_emotions)))
        
        accuracy = accuracy_score(true_emotions, pred_emotions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_emotions, 
            pred_emotions, 
            labels=unique_labels,  # Use only actual labels
            average='weighted',
            zero_division=1  # Set to 1 instead of 0
        )
        
        # Class-level metrics
        class_metrics = precision_recall_fscore_support(
            true_emotions,
            pred_emotions, 
            labels=self.emotion_classes,
            zero_division=1  # Set to 1 instead of 0
        )
        
        per_class_metrics = {}
        for i, emotion in enumerate(self.emotion_classes):
            per_class_metrics[emotion] = {
                "precision": class_metrics[0][i],
                "recall": class_metrics[1][i],
                "f1": class_metrics[2][i],
                "support": class_metrics[3][i]
            }
            
        return {
            "overall": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "per_class": per_class_metrics
        }

    def _save_results(self, predictions, true_emotions, pred_emotions, metrics):
        """Save evaluation results"""
        # Get Hydra's runtime output directory
        hydra_config = HydraConfig.get()
        
        # Use Hydra's output directory
        results_dir = Path(hydra_config.runtime.output_dir) / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj]
            return obj

        # Get model configuration info
        config_info = {
            "data": {
                "n_samples": self.cfg.data.n_samples,
                "dataset": self.cfg.data.name
            },
            "model": {
                "provider": self.cfg.model.provider,
                "use_rag": self.cfg.model.use_rag,
                "embedding": {
                    "provider": self.cfg.model.provider
                },
                "chat": {
                    "provider": self.cfg.model.provider
                }
            },
            "metrics": convert_to_serializable(metrics),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        # Safely add model information based on provider
        if self.cfg.model.provider == "lmstudio":
            if self.cfg.model.use_rag:
                embedding_info = self.analyzer.embedding_info
                if isinstance(embedding_info, dict) and "model" in embedding_info:
                    config_info["model"]["embedding"]["model_id"] = embedding_info["model"].get("id", "unknown")
            
            chat_model_info = self.analyzer.chat_model_info
            if isinstance(chat_model_info, dict) and "model" in chat_model_info:
                config_info["model"]["chat"]["model_id"] = chat_model_info["model"].get("id", "unknown")
        else:  # openai
            if self.cfg.model.use_rag:
                config_info["model"]["embedding"]["model_id"] = self.cfg.model.openai.embedding_model
            config_info["model"]["chat"]["model_id"] = self.cfg.model.openai.chat_model_name

        # Save detailed results
        with open(results_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)

        # Save evaluation report
        report = self._create_evaluation_report(predictions, true_emotions, pred_emotions, metrics)
        with open(results_dir / "report.txt", "w", encoding="utf-8") as f:
            f.write(report)

    def _create_evaluation_report(self, predictions, true_emotions, pred_emotions, metrics):
        """Create detailed evaluation report"""
        report = []
        report.append("Emotion Classification Evaluation Report")
        report.append("=" * 50)
        
        # Overall metrics
        report.append("\nOverall Metrics:")
        report.append("-" * 20)
        report.append(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
        report.append(f"Precision: {metrics['overall']['precision']:.4f}")
        report.append(f"Recall: {metrics['overall']['recall']:.4f}")
        report.append(f"F1 Score: {metrics['overall']['f1']:.4f}")
        
        # Per-class metrics
        report.append("\nPer-class Metrics:")
        report.append("-" * 20)
        for emotion, scores in metrics['per_class'].items():
            report.append(f"\n{emotion}:")
            report.append(f"  Precision: {scores['precision']:.4f}")
            report.append(f"  Recall: {scores['recall']:.4f}")
            report.append(f"  F1 Score: {scores['f1']:.4f}")
            report.append(f"  Support: {scores['support']}")
        
        return "\n".join(report) 