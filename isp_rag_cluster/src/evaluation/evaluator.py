from .metrics import MetricsManager
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

class Evaluator:
    def __init__(self, analyzer, cfg):
        self.analyzer = analyzer
        self.cfg = cfg
        self.metrics_manager = MetricsManager(cfg)
        self.emotion_classes = cfg.emotions.classes

    def evaluate(self, eval_df):
        predictions = []
        true_emotions = []
        pred_emotions = []

        for _, row in tqdm(eval_df.iterrows(), desc="Evaluating"):
            result = self._evaluate_single(row)
            predictions.append(result)
            true_emotions.append(result["true_emotion"])
            pred_emotions.append(result["predicted_emotion"])

        # 메트릭 계산
        metrics = self._calculate_metrics(true_emotions, pred_emotions)
        
        # 결과 저장
        self._save_results(predictions, true_emotions, pred_emotions, metrics)
        
        return metrics

    def _evaluate_single(self, row):
        text = row[self.cfg.data.column_mapping.text]
        # true_emotion이 숫자(1-7)로 들어오므로 감정 이름으로 변환 필요
        emotion_id = str(row[self.cfg.data.column_mapping.emotion])
        true_emotion = self.cfg.emotions.classes[int(emotion_id) - 1]  # 1-based to 0-based indexing
        
        # analyzer.analyze_emotion() 메서드를 통해 RAG 사용 여부 제어
        pred_emotion = self.analyzer.analyze_emotion(text)

        # RAG 사용 시에만 retrieved_context 포함
        result = {
            "text": text,
            "true_emotion": true_emotion,
            "predicted_emotion": pred_emotion,
        }
        
        if self.cfg.model.use_rag:
            result["retrieved_context"] = self.analyzer.retriever.retrieve(text)
        
        return result

    def _calculate_metrics(self, true_emotions, pred_emotions):
        # 모든 경고 메시지 무시
        warnings.filterwarnings('ignore')
        
        # 실제로 나타난 레이블만 사용
        unique_labels = sorted(list(set(true_emotions) | set(pred_emotions)))
        
        accuracy = accuracy_score(true_emotions, pred_emotions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_emotions, 
            pred_emotions, 
            labels=unique_labels,  # 실제 나타난 레이블만 사용
            average='weighted',
            zero_division=1  # 0 대신 1로 설정
        )
        
        # 클래스별 메트릭
        class_metrics = precision_recall_fscore_support(
            true_emotions,
            pred_emotions, 
            labels=self.emotion_classes,
            zero_division=1  # 0 대신 1로 설정
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
        # 설정 정보 저장
        config_info = {
            "n_samples": self.cfg.data.n_samples,
            "model_provider": self.cfg.model.provider,
            "embedding_info": self.analyzer.embedding_info,
            "data_split": {
                "train_size": self.cfg.data.train_size,
                "val_size": self.cfg.data.val_size
            }
        }
        self.metrics_manager.save_config(config_info)
        
        # 메트릭 저장
        self.metrics_manager.save_metrics(metrics)
        
        # 예측 결과 저장
        self.metrics_manager.save_predictions(predictions)
        
        # 분류 보고서 저장
        self.metrics_manager.save_classification_report(
            true_emotions, pred_emotions, self.emotion_classes
        )
        
        # 혼동 행렬 저장
        self.metrics_manager.save_confusion_matrix(
            true_emotions, pred_emotions, self.emotion_classes
        ) 