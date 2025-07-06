import os
# Set OpenMP environment variables to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Intel OpenMP 충돌 방지
os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP 스레드 수 제한

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter
import json
import umap
from tqdm import tqdm
from .deep_clustering import DeepEmbeddedClustering as DEC  # DEC 클래스 import
from sklearn.metrics import confusion_matrix
import shutil
from hydra.core.hydra_config import HydraConfig

class EmbeddingAnalyzer:
    def __init__(self, cfg):
        # OpenMP 설정 추가
        if os.name != 'nt':  # Windows가 아닌 경우
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        self.cfg = cfg
        #self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(cfg.general.outputs) / "embedding_analysis" 
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze(self, embeddings: np.ndarray, labels: List[str]):
        """Analyze embeddings using selected method"""
        if self.cfg.analysis.embedding.method == "traditional":
            # Traditional 방식: 차원 축소 + 클러스터링
            print("\nPerforming dimensionality reduction...")
            reduced_data = self._reduce_dimensions(embeddings)
            
            print("\nPerforming clustering...")
            clustering_results = self._perform_clustering(reduced_data, labels)
            
            # 결과 저장
            self._save_results(reduced_data, clustering_results, labels)
            return clustering_results
        elif self.cfg.analysis.embedding.method == "dec":
            # DEC 초기화 및 학습
            dec = DEC(
                input_dim=embeddings.shape[1],
                n_clusters=len(set(labels)),
                encoder_dims=self.cfg.analysis.embedding.dec.hidden_dims,
                pretrain_epochs=self.cfg.analysis.embedding.dec.pretrain_epochs,
                clustering_epochs=self.cfg.analysis.embedding.dec.clustering_epochs,
                batch_size=self.cfg.analysis.embedding.dec.batch_size,
                update_interval=self.cfg.analysis.embedding.dec.update_interval,
                tol=self.cfg.analysis.embedding.dec.tol,
                learning_rate=self.cfg.analysis.embedding.dec.learning_rate
            )
            
            # 학습 및 클러스터링
            print("\nTraining DEC model...")
            cluster_labels = dec.fit_predict(embeddings)
            
            # 결과 평가
            results = self._evaluate_dec_results(
                embeddings, cluster_labels, labels, 
                len(set(labels))
            )
        
        # 결과 저장
            reduced_data = dec.encoder.predict(embeddings)  # 인코더로 차원 축소
            self._save_results(reduced_data, results, labels)
            
            return results
        else:
            raise ValueError(f"Unknown method: {self.cfg.analysis.embedding.method}")
        
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """차원 축소 수행"""
        reducer_type = self.cfg.analysis.embedding.traditional.reducer.type
        
        if reducer_type == "pca":
            reducer = PCA(
                n_components=self.cfg.analysis.embedding.traditional.reducer.pca.n_components
            )
        elif reducer_type == "umap":
            reducer = umap.UMAP(
                n_neighbors=self.cfg.analysis.embedding.traditional.reducer.umap.n_neighbors,
                min_dist=self.cfg.analysis.embedding.traditional.reducer.umap.min_dist,
                n_components=self.cfg.analysis.embedding.traditional.reducer.umap.n_components,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown reducer type: {reducer_type}")
            
        return reducer.fit_transform(embeddings)
    
    def _perform_clustering(self, data: np.ndarray, true_labels: List[str]) -> Dict[str, Any]:
        results = {
            "metrics": {
                "intrinsic": {},
                "extrinsic": {}
            },
            "best_k": None,
            "best_score": 0.0,
            "clustering_type": "traditional",
            "cluster_labels": None,  # 클러스터 레이블 추가
            "best_k_criterion": {    # best_k_criterion 추가
                "type": "extrinsic",
                "metric": "normalized_mutual_info"
            }
        }
        
        k_range = self.cfg.analysis.embedding.traditional.clustering.k_range
        
        # 클러스터링 수행
        best_score = -np.inf
        best_labels = None
        
        for k in k_range:
            if self.cfg.analysis.embedding.traditional.clustering.type == "kmeans":
                clusterer = KMeans(
                    n_clusters=k,
                    n_init=self.cfg.analysis.embedding.traditional.clustering.kmeans.n_init,
                    random_state=self.cfg.analysis.embedding.traditional.clustering.kmeans.random_state
                )
            else:  # gmm
                clusterer = GaussianMixture(
                    n_components=k,
                    covariance_type=self.cfg.analysis.embedding.traditional.clustering.gmm.covariance_type,
                    random_state=self.cfg.analysis.embedding.traditional.clustering.gmm.random_state
                )
            
            cluster_labels = clusterer.fit_predict(data)
            
            # 내부 평가
            results["metrics"]["intrinsic"][k] = {
                "silhouette": silhouette_score(data, cluster_labels),
                "calinski_harabasz": calinski_harabasz_score(data, cluster_labels),
                "davies_bouldin_inverted": -davies_bouldin_score(data, cluster_labels)
            }
            
            # 외부 평가
            results["metrics"]["extrinsic"][k] = {
                "adjusted_rand": adjusted_rand_score(true_labels, cluster_labels),
                "normalized_mutual_info": normalized_mutual_info_score(true_labels, cluster_labels)
            }
            
            # 최고 점수 갱신
            current_score = results["metrics"]["extrinsic"][k]["normalized_mutual_info"]
            if current_score > best_score:
                best_score = current_score
                best_labels = cluster_labels
                results["best_k"] = k
                results["best_score"] = current_score
        
        # 최종 클러스터 레이블 저장
        results["cluster_labels"] = best_labels
        
        # Knee/Elbow 포인트 찾기
        results["knee_points"] = self._find_knee_points(results["metrics"])
        
        return results
    
    def _find_knee_points(self, metrics: Dict) -> Dict:
        knee_points = {}
        
        # 내부 메트릭의 평균으로 knee point 찾기
        intrinsic_means = []
        for k in metrics["intrinsic"].keys():
            scores = metrics["intrinsic"][k]
            # 모든 점수를 정규화하고 평균
            normalized_scores = []
            for metric, score in scores.items():
                values = [metrics["intrinsic"][k2][metric] for k2 in metrics["intrinsic"].keys()]
                normalized = (score - min(values)) / (max(values) - min(values))
                normalized_scores.append(normalized)
            intrinsic_means.append(np.mean(normalized_scores))
        
        knee_points["intrinsic"] = KneeLocator(
            list(metrics["intrinsic"].keys()),
            intrinsic_means,
            curve="convex",
            direction="increasing"
        ).knee
        
        # 외부 메트릭도 동일하게 처리
        extrinsic_means = []
        for k in metrics["extrinsic"].keys():
            scores = metrics["extrinsic"][k]
            normalized_scores = []
            for metric, score in scores.items():
                values = [metrics["extrinsic"][k2][metric] for k2 in metrics["extrinsic"].keys()]
                normalized = (score - min(values)) / (max(values) - min(values))
                normalized_scores.append(normalized)
            extrinsic_means.append(np.mean(normalized_scores))
        
        knee_points["extrinsic"] = KneeLocator(
            list(metrics["extrinsic"].keys()),
            extrinsic_means,
            curve="convex",
            direction="increasing"
        ).knee
        
        return knee_points

    def _save_results(self, reduced_data: np.ndarray, clustering_results: Dict[str, Any], true_labels: List[str]):
        """결과 저장"""
        # Save config by copying hydra config
        from hydra.core.hydra_config import HydraConfig
        
        # Create config directory
        config_dir = self.results_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Get hydra config info
        hydra_config = HydraConfig.get()
        config_name = hydra_config.job.config_name  # 실제 사용된 config 파일 이름 가져오기
        
        # Get source config files from hydra output directory
        hydra_output_dir = Path(hydra_config.runtime.output_dir)
        
        # Copy the actual config file being used
        config_files = [
            (f"{config_name}.yaml", f"{config_name}.yaml"),  # 메인 config 파일
            (".hydra/hydra.yaml", "hydra.yaml"),             # hydra config
            (".hydra/overrides.yaml", "overrides.yaml")      # override 설정
        ]
        
        for src_name, dst_name in config_files:
            src_path = hydra_output_dir / src_name
            if src_path.exists():
                dst_path = config_dir / dst_name
                print(f"Copying config file: {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Config file not found: {src_path}")

        # Save clustering summary
        summary = {
            "best_k": int(clustering_results["best_k"]),
            "knee_points": self._convert_to_serializable(clustering_results.get("knee_points", {})),
            "best_scores": {
                "intrinsic": self._convert_to_serializable(clustering_results["metrics"]["intrinsic"]),
                "extrinsic": self._convert_to_serializable(clustering_results["metrics"]["extrinsic"])
            }
        }
        
        with open(self.results_dir / "clustering_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Plot clustering results
        if clustering_results["clustering_type"] == "traditional":
            self._plot_metrics(
                clustering_results["metrics"],
                clustering_results.get("knee_points", {})
            )
        
        # Plot cluster visualization for both methods
        self._plot_clusters_2d(
            reduced_data, 
            clustering_results["cluster_labels"],
            true_labels,
            "Clustering Results"
        )
        
        # Plot both confusion matrix and class-cluster agreement
        self._plot_confusion_matrices(
            true_labels,
            clustering_results["cluster_labels"]
        )

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

    def _save_numerical_results(self, clustering_results: Dict[str, Any]):
        """수치 결과 저장"""
        # 1. 원본 메트릭 값 저장
        metrics_df = pd.DataFrame()
        
        # Intrinsic 메트릭
        for k, scores in clustering_results["metrics"]["intrinsic"].items():
            row = {"k": k, "metric_type": "intrinsic"}
            row.update({f"{name}_raw": score for name, score in scores.items()})
            metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        
        # Extrinsic 메트릭
        for k, scores in clustering_results["metrics"]["extrinsic"].items():
            row = {"k": k, "metric_type": "extrinsic"}
            row.update({f"{name}_raw": score for name, score in scores.items()})
            metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        
        # 2. 정규화된 메트릭 값 추가
        for metric_type in ["intrinsic", "extrinsic"]:
            metrics = clustering_results["metrics"][metric_type]
            for metric_name in metrics[list(metrics.keys())[0]].keys():
                values = [metrics[k][metric_name] for k in metrics.keys()]
                normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
                
                # 정규화된 값 추가
                for i, k in enumerate(metrics.keys()):
                    mask = (metrics_df["k"] == k) & (metrics_df["metric_type"] == metric_type)
                    metrics_df.loc[mask, f"{metric_name}_normalized"] = normalized_values[i]
        
        # CSV로 저장
        metrics_df.to_csv(self.results_dir / "clustering_metrics.csv", index=False)
        
        # 3. 요약 정보 JSON으로 저장
        summary = {
            "best_k": clustering_results["best_k"],
            "knee_points": self._convert_to_serializable(clustering_results["knee_points"]),
            "best_scores": {
                "intrinsic": {
                    name: self._convert_to_serializable(
                        max(scores[name] for scores in clustering_results["metrics"]["intrinsic"].values())
                    )
                    for name in clustering_results["metrics"]["intrinsic"][list(clustering_results["metrics"]["intrinsic"].keys())[0]].keys()
                },
                "extrinsic": {
                    name: self._convert_to_serializable(
                        max(scores[name] for scores in clustering_results["metrics"]["extrinsic"].values())
                    )
                    for name in clustering_results["metrics"]["extrinsic"][list(clustering_results["metrics"]["extrinsic"].keys())[0]].keys()
                }
            }
        }
        
        with open(self.results_dir / "clustering_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _plot_dec_metrics(self, metrics: Dict[str, Dict[int, Dict[str, float]]], best_k: int):
        """DEC 메트릭 시각화"""
        plt.figure(figsize=(10, 6))
        
        # Plot intrinsic and extrinsic metrics as bar charts
        metric_types = ["intrinsic", "extrinsic"]
        n_metrics = len(metric_types)
        
        for i, metric_type in enumerate(metric_types):
            plt.subplot(1, 2, i+1)
            metric_values = metrics[metric_type][best_k]
            
            # Create bar plot
            plt.bar(range(len(metric_values)), 
                    list(metric_values.values()),
                    tick_label=list(metric_values.keys()))
            
            plt.title(f'{metric_type.capitalize()} Metrics')
            plt.xticks(rotation=45)
            plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "dec_metrics.png")
        plt.close()

    def _plot_metrics(self, metrics: Dict[str, Dict[int, Dict[str, float]]], 
                      knee_points: Dict[str, float]):
        """메트릭 변화 시각화 (traditional method only)"""
        # 개별 메트릭 그룹 플롯
        self._plot_metric_group(
            metrics["intrinsic"], 
            "Intrinsic Metrics", 
            "intrinsic_metrics.png",
            knee_points["intrinsic"]
        )
        
        self._plot_metric_group(
            metrics["extrinsic"], 
            "Extrinsic Metrics", 
            "extrinsic_metrics.png",
            knee_points["extrinsic"]
        )
        
        # 평균 비교 플롯
        plt.figure(figsize=(12, 6))
        k_values = list(metrics["intrinsic"].keys())
        
        # 각 그룹의 평균과 표준오차 계산
        for metric_type in ["intrinsic", "extrinsic"]:
            means = []
            errors = []
            for k in k_values:
                scores = list(metrics[metric_type][k].values())
                normalized_scores = []
                for i, score in enumerate(scores):
                    values = [metrics[metric_type][k2][list(metrics[metric_type][k2].keys())[i]] 
                             for k2 in k_values]
                    normalized = (score - min(values)) / (max(values) - min(values))
                    normalized_scores.append(normalized)
                means.append(np.mean(normalized_scores))
                errors.append(np.std(normalized_scores) / np.sqrt(len(normalized_scores)))
            
            plt.errorbar(k_values, means, yerr=errors, 
                        label=f'{metric_type.capitalize()} Metrics',
                        marker='o', capsize=5)
            
            # Knee point 표시
            if knee_points[metric_type]:
                plt.axvline(x=knee_points[metric_type], 
                           color='gray', linestyle='--', alpha=0.5)
                plt.text(knee_points[metric_type], plt.ylim()[0], 
                        f'{metric_type} knee', rotation=90)
        
        plt.title('Comparison of Metric Groups')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Normalized Score (Mean ± SE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.results_dir / "metrics_comparison.png")
        plt.close()

    def _normalize_values(self, values):
        """안전한 정규화 수행"""
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:
            return np.ones_like(values)  # 모든 값이 같으면 1로 정규화
        return (values - min_val) / (max_val - min_val)

    def _plot_metric_group(self, metrics: Dict[int, Dict[str, float]], 
                          title: str, filename: str, knee_point: float = None):
        """메트릭 그룹별 시각화"""
        plt.figure(figsize=(10, 6))
        
        # 데이터 정규화 및 플로팅
        for metric_name in metrics[list(metrics.keys())[0]].keys():
            values = [metrics[k][metric_name] for k in metrics.keys()]
            normalized_values = self._normalize_values(values)
            plt.plot(list(metrics.keys()), normalized_values, 
                    marker='o', label=metric_name)
        
        plt.title(f'{title} (Normalized)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Score (Normalized)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.results_dir / filename)
        plt.close()

    def _evaluate_dec_results(self, embeddings: np.ndarray, cluster_labels: np.ndarray, 
                         true_labels: List[str], n_clusters: int) -> Dict[str, Any]:
        """DEC 결과 평가"""
        results = {
            "clustering_type": "dec",
            "best_k": n_clusters,
            "cluster_labels": cluster_labels,
            "best_k_criterion": {
                "type": "fixed",
                "metric": "n_classes"
            },
            "metrics": {
                "intrinsic": {
                    "silhouette": silhouette_score(embeddings, cluster_labels),
                    "calinski_harabasz": calinski_harabasz_score(embeddings, cluster_labels),
                    "davies_bouldin_inverted": -davies_bouldin_score(embeddings, cluster_labels)
                },
                "extrinsic": {
                    "adjusted_rand": adjusted_rand_score(true_labels, cluster_labels),
                    "normalized_mutual_info": normalized_mutual_info_score(true_labels, cluster_labels)
                }
            }
        }
        
        # Best score for DEC is NMI
        results["best_score"] = results["metrics"]["extrinsic"]["normalized_mutual_info"]
        
        return results

    def _convert_labels_to_emotions(self, labels):
        """라벨을 감정 이름으로 변환"""
        emotion_classes = self.cfg.emotions.classes
        
        # 이미 감정 이름이면 그대로 반환
        if isinstance(labels[0], str) and labels[0] in emotion_classes:
            return labels
        
        # 숫자를 0-based index로 변환 후 감정 이름으로 매핑
        return [emotion_classes[int(label) - 1] for label in labels]

    def _plot_clusters_2d(self, data_2d: np.ndarray, cluster_labels: np.ndarray, 
                         true_labels: List[str], title: str):
        """2D 클러스터링 결과 시각화"""
        plt.figure(figsize=(12, 5))
        
        # Plot predicted clusters
        plt.subplot(121)
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                             c=cluster_labels, 
                             cmap=self.cfg.analysis.embedding.visualization.scatter.cmap,
                             alpha=self.cfg.analysis.embedding.visualization.scatter.alpha)
        plt.title(f"{title} (Predicted)")
        plt.colorbar(scatter, label="Cluster")
        
        # Plot true labels
        plt.subplot(122)
        emotion_classes = self.cfg.emotions.classes
        true_emotions = self._convert_labels_to_emotions(true_labels)
        
        # 감정 이름을 숫자로 매핑
        emotion_to_int = {emotion: i for i, emotion in enumerate(emotion_classes)}
        true_labels_int = [emotion_to_int[emotion] for emotion in true_emotions]
        
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                             c=true_labels_int,
                             cmap=self.cfg.analysis.embedding.visualization.scatter.cmap,
                             alpha=self.cfg.analysis.embedding.visualization.scatter.alpha)
        plt.title(f"{title} (True Classes)")
        plt.colorbar(scatter, ticks=range(len(emotion_classes)), 
                    label="Emotion Classes").set_ticklabels(emotion_classes)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "clustering_visualization.png")
        plt.close()

    def _plot_confusion_matrices(self, true_labels: List[str], pred_labels: np.ndarray):
        """클래스-클러스터 일치도 시각화"""
        emotion_classes = self.cfg.emotions.classes
        true_emotions = self._convert_labels_to_emotions(true_labels)
        unique_clusters = sorted(list(set(pred_labels)))
        
        # Class-Cluster Agreement Matrix
        agreement_matrix = np.zeros((len(emotion_classes), len(unique_clusters)))
        
        # Calculate class-cluster distribution
        for i, emotion in enumerate(emotion_classes):
            for j, cluster in enumerate(unique_clusters):
                agreement_matrix[i, j] = np.sum(
                    (np.array(true_emotions) == emotion) & (pred_labels == cluster)
                )
        
        # Normalize by column
        agreement_matrix = agreement_matrix / agreement_matrix.sum(axis=0, keepdims=True)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(agreement_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap=self.cfg.analysis.embedding.visualization.heatmap.cmap,
                    xticklabels=[f'Cluster {i}' for i in range(len(unique_clusters))],
                    yticklabels=emotion_classes)
        plt.title('Class-Cluster Agreement')
        plt.ylabel('Emotion Class')
        plt.xlabel('Cluster')
        plt.tight_layout()
        plt.savefig(self.results_dir / "class_cluster_agreement.png")
        plt.close() 