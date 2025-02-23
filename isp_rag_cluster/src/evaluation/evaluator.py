from .metrics import MetricsManager
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
from datetime import datetime

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
                    "provider": self.cfg.model.provider,
                    "base_url": self.cfg.model.lmstudio.base_url if self.cfg.model.provider == "lmstudio" else None
                },
                "chat": {
                    "provider": self.cfg.model.provider
                }
            },
            "retriever": {
                "k": self.cfg.model.retriever.k,
                "score_threshold": self.cfg.model.retriever.score_threshold,
                "search_type": self.cfg.model.retriever.search_type
            } if self.cfg.model.use_rag else None,
            "metrics": metrics,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        # Safely add model information
        embedding_info = self.analyzer.embedding_info
        if isinstance(embedding_info, dict):
            # Handle nested model info
            model_info = embedding_info.get("model", {})
            if isinstance(model_info, dict):
                config_info["model"]["embedding"]["model_id"] = model_info.get("id", "unknown")
            else:
                config_info["model"]["embedding"]["model_id"] = str(model_info)
        else:
            config_info["model"]["embedding"]["model_id"] = str(embedding_info)

        chat_model_info = self.analyzer.chat_model_info
        if isinstance(chat_model_info, dict):
            model_info = chat_model_info.get("model", {})
            if isinstance(model_info, dict):
                config_info["model"]["chat"]["model_id"] = model_info.get("id", "unknown")
            else:
                config_info["model"]["chat"]["model_id"] = str(model_info)
        else:
            config_info["model"]["chat"]["model_id"] = str(chat_model_info)
        
        # Save config
        self.metrics_manager.save_config(config_info)
        
        # Save classification report
        report = classification_report(true_emotions, pred_emotions)
        with open(self.metrics_manager.analysis_dir / "classification_report.txt", "w") as f:
            f.write(report)
        
        # Save predictions
        self.metrics_manager.save_predictions(predictions)
        
        # Save confusion matrix
        self.metrics_manager.save_confusion_matrix(
            true_emotions, pred_emotions, self.emotion_classes
        ) 