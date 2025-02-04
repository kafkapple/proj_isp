from typing import Dict, List
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import logging
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class EmotionMetrics:
    def __init__(self, num_classes: int, class_names: List[str], config: DictConfig):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.config = config
        self.current_epoch = 0
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []  # 확률값 저장을 위한 리스트 추가
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        if isinstance(preds, torch.Tensor):
            # 확률값 저장 (softmax 적용)
            probs = torch.nn.functional.softmax(preds, dim=1)
            self.all_probs.append(probs.detach().cpu().numpy())  # extend 대신 append 사용
            
            # 예측값 저장
            if preds.dim() > 1:
                preds = torch.argmax(preds, dim=1)
            preds = preds.cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        self.all_preds.append(preds)  # extend 대신 append 사용
        self.all_labels.append(labels)
    
    def get_classification_report(self) -> str:
        """return classification report for current epoch"""
        if not self.all_labels or not self.all_preds:
            return "No predictions available"
            
        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0
        )
        
        # print to console
        logging.info("\nClassification Report:")
        logging.info("="*50)
        logging.info(report)
        logging.info("="*50)
        
        return report
    
    def compute(self, prefix: str = "") -> Dict[str, float]:
        """Compute all metrics and save visualizations"""
        if not self.all_labels or not self.all_preds:
            return {}
        
        # Concatenate all arrays
        y_true = np.concatenate(self.all_labels)
        y_pred = np.concatenate(self.all_preds)
        y_score = np.concatenate(self.all_probs)
        
        # 기본 메트릭스 계산
        metrics = {
            f"{prefix}/accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}/macro_f1": f1_score(y_true, y_pred, average='macro'),
            f"{prefix}/weighted_f1": f1_score(y_true, y_pred, average='weighted')
        }
        
        # 클래스별 F1 점수
        class_f1 = f1_score(y_true, y_pred, average=None)
        for i, score in enumerate(class_f1):
            metrics[f"{prefix}/f1_{self.class_names[i]}"] = score
        
        # Confusion Matrix 생성 및 저장
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = self._plot_confusion_matrix(cm, self.class_names)
        
        # ROC Curve 생성 및 저장
        fig_roc = self._plot_roc_curve(y_true, y_score, self.class_names)
        
        # PR Curve 생성 및 저장
        fig_pr = self._plot_pr_curve(y_true, y_score, self.class_names)
        
        # Classification Report 생성 및 파싱
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True  # dictionary 형태로 반환
        )
        
        # Text 형태의 report도 생성 (파일 저장용)
        report_text = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        
        # WandB 로깅
        if wandb.run is not None:
            # 기존 로깅
            wandb.log({
                f"{prefix}/confusion_matrix": wandb.Image(fig_cm),
                f"{prefix}/roc_curve": wandb.Image(fig_roc),
                f"{prefix}/pr_curve": wandb.Image(fig_pr),
            }, step=self.current_epoch)
            
            # Classification Report를 테이블로 변환하여 로깅
            report_table_data = []
            
            # 각 클래스별 메트릭 추가
            for class_name in self.class_names:
                class_metrics = report[class_name]
                report_table_data.append([
                    class_name,
                    class_metrics['precision'],
                    class_metrics['recall'],
                    class_metrics['f1-score'],
                    class_metrics['support']
                ])
            
            # macro avg, weighted avg 추가
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    report_table_data.append([
                        avg_type,
                        report[avg_type]['precision'],
                        report[avg_type]['recall'],
                        report[avg_type]['f1-score'],
                        report[avg_type]['support']
                    ])
            
            # WandB 테이블 생성 및 로깅
            wandb.log({
                f"{prefix}/classification_report": wandb.Table(
                    columns=["Class", "Precision", "Recall", "F1-score", "Support"],
                    data=report_table_data
                ),
                # 기존 메트릭 테이블도 유지
                f"{prefix}/metrics": wandb.Table(
                    columns=["Metric", "Value"],
                    data=[[k, v] for k, v in metrics.items()]
                )
            }, step=self.current_epoch)
        
        # 로컬 파일 저장
        output_dir = Path(self.config.dirs.outputs) / "plots" / str(self.current_epoch)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Text 형태의 report 저장
        with open(output_dir / f"{prefix}_classification_report.txt", "w") as f:
            f.write(report_text)
        
        # Close figures to free memory
        plt.close(fig_cm)
        plt.close(fig_roc)
        plt.close(fig_pr)
        
        print(f"\n{prefix.upper()} Classification Report - Epoch {self.current_epoch}:")
        print("="*50)
        print(report_text)
        print("="*50)
        
        return metrics

    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - Epoch {self.current_epoch}')
        return fig

    def _plot_roc_curve(self, y_true, y_score, class_names):
        """Plot ROC curve"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert tensors to numpy arrays
        y_true_np = y_true
        y_score_np = y_score
        
        for i in range(len(class_names)):
            # Binary classification for each class
            y_true_binary = (y_true_np == i).astype(int)
            y_score_class = y_score_np[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_class)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - Epoch {self.current_epoch}')
        ax.legend(loc='lower right')
        return fig

    def _plot_pr_curve(self, y_true, y_score, class_names):
        """Plot Precision-Recall curve"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert tensors to numpy arrays
        y_true_np = y_true
        y_score_np = y_score
        
        for i in range(len(class_names)):
            # Binary classification for each class
            y_true_binary = (y_true_np == i).astype(int)
            y_score_class = y_score_np[:, i]
            
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score_class)
            ap = average_precision_score(y_true_binary, y_score_class)
            ax.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.2f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - Epoch {self.current_epoch}')
        ax.legend(loc='lower left')
        return fig

    def set_epoch(self, epoch: int):
        """Set current epoch number"""
        self.current_epoch = epoch 