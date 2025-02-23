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

class EmbeddingAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / "embedding_analysis" / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze(self, embeddings: np.ndarray, labels: List[str]):
        """임베딩 분석 수행"""
        if self.cfg.analysis.embedding.method == "traditional":
            # 기존 방식: 차원 축소 + 클러스터링
            reduced_data = self._reduce_dimensions(embeddings)
            clustering_results = self._perform_clustering(reduced_data, labels)
        else:  # dec
            # DEC 방식
            from .deep_clustering import DeepEmbeddedClustering
            dec = DeepEmbeddedClustering(self.cfg, input_dim=embeddings.shape[1])
            
            # 사전 학습
            print("Pretraining autoencoder...")
            dec.pretrain(embeddings)
            
            # 클러스터링
            print("\nPerforming deep clustering...")
            best_k = self._find_best_k_for_dec(dec, embeddings, labels)
            dec_results = dec.cluster(embeddings, best_k)
            
            # 결과 변환
            reduced_data = dec_results["latent_features"]
            clustering_results = self._evaluate_dec_results(
                dec_results["cluster_labels"], 
                labels, 
                best_k
            )
        
        # 결과 저장
        self._save_results(reduced_data, clustering_results, labels)
        return clustering_results
        
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
            "clustering_type": self.cfg.analysis.embedding.traditional.clustering.type
        }
        
        k_range = self.cfg.analysis.embedding.traditional.clustering.k_range
        
        # 클러스터링 수행
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
            intrinsic_scores = {
                "silhouette": silhouette_score(data, cluster_labels),
                "calinski_harabasz": calinski_harabasz_score(data, cluster_labels),
                "davies_bouldin_inverted": -davies_bouldin_score(data, cluster_labels)
            }
            
            # 외부 평가
            extrinsic_scores = {
                "adjusted_rand": adjusted_rand_score(true_labels, cluster_labels),
                "normalized_mutual_info": normalized_mutual_info_score(true_labels, cluster_labels)
            }
            
            results["metrics"]["intrinsic"][k] = intrinsic_scores
            results["metrics"]["extrinsic"][k] = extrinsic_scores
        
        # Knee/Elbow 포인트 찾기
        results["knee_points"] = self._find_knee_points(results["metrics"])
        
        # 설정에서 best k 선택 기준 가져오기
        criterion = self.cfg.analysis.embedding.traditional.clustering.best_k_criterion
        metric_type = criterion.type
        metric_name = criterion.metric
        
        if metric_type == "combined":
            # 모든 메트릭의 정규화된 평균 계산
            combined_scores = []
            for k in k_range:
                # 내부 메트릭 정규화 및 평균
                intrinsic_scores = []
                for name, score in results["metrics"]["intrinsic"][k].items():
                    values = [results["metrics"]["intrinsic"][k2][name] for k2 in k_range]
                    if name == "davies_bouldin_inverted":
                        normalized = (score - min(values)) / (max(values) - min(values))
                    else:
                        normalized = (score - min(values)) / (max(values) - min(values))
                    intrinsic_scores.append(normalized)
                
                # 외부 메트릭 정규화 및 평균
                extrinsic_scores = []
                for name, score in results["metrics"]["extrinsic"][k].items():
                    values = [results["metrics"]["extrinsic"][k2][name] for k2 in k_range]
                    normalized = (score - min(values)) / (max(values) - min(values))
                    extrinsic_scores.append(normalized)
                
                # 전체 평균 계산
                combined_score = (np.mean(intrinsic_scores) + np.mean(extrinsic_scores)) / 2
                combined_scores.append(combined_score)
            
            best_k_idx = np.argmax(combined_scores)
            results["best_k"] = list(k_range)[best_k_idx]
            results["best_score"] = combined_scores[best_k_idx]
            
        else:
            # 기존 로직 (단일 메트릭 기반)
            scores = [scores[metric_name] 
                     for scores in results["metrics"][metric_type].values()]
            
            if metric_name == "davies_bouldin_inverted":
                best_k_idx = np.argmin(scores)
            else:
                best_k_idx = np.argmax(scores)
            
            results["best_k"] = list(k_range)[best_k_idx]
            results["best_score"] = scores[best_k_idx]
        
        results["best_k_criterion"] = {
            "type": metric_type,
            "metric": metric_name
        }
        
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
        # 시각화 설정
        viz_cfg = self.cfg.analysis.embedding.visualization
        
        # 최적의 k로 클러스터링
        best_k = clustering_results["best_k"]
        
        # 클러스터링 수행
        if self.cfg.analysis.embedding.traditional.clustering.type == "kmeans":
            clusterer = KMeans(
                n_clusters=best_k,
                n_init=self.cfg.analysis.embedding.traditional.clustering.kmeans.n_init,
                random_state=self.cfg.analysis.embedding.traditional.clustering.kmeans.random_state
            )
        else:  # gmm
            clusterer = GaussianMixture(
                n_components=best_k,
                covariance_type=self.cfg.analysis.embedding.traditional.clustering.gmm.covariance_type,
                random_state=self.cfg.analysis.embedding.traditional.clustering.gmm.random_state
            )
        
        cluster_labels = clusterer.fit_predict(reduced_data)
        
        # 1. 3개의 서브플롯 생성
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Emotion class별 scatter plot
        unique_emotions = sorted(set(true_labels))
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_emotions)))  # Set2 팔레트 사용
        for emotion, color in zip(unique_emotions, colors):
            mask = np.array(true_labels) == emotion
            ax1.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                       c=[color], label=emotion, 
                       alpha=viz_cfg.scatter.alpha)
        ax1.set_title('Emotion Classes')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Clustering 결과 scatter plot
        scatter = ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                            c=cluster_labels, 
                            cmap=viz_cfg.scatter.cmap,
                            alpha=viz_cfg.scatter.alpha)
        ax2.set_title(f'Clustering Results (k={best_k})')
        legend1 = ax2.legend(*scatter.legend_elements(),
                           title="Clusters",
                           bbox_to_anchor=(1.05, 1), 
                           loc='upper left')
        ax2.add_artist(legend1)
        
        # Cluster-Class 관계 heatmap
        confusion_matrix = np.zeros((len(unique_emotions), best_k))
        for i, emotion in enumerate(unique_emotions):
            for j in range(best_k):
                mask = (np.array(true_labels) == emotion) & (cluster_labels == j)
                confusion_matrix[i, j] = mask.sum()
        
        # 정규화
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        
        sns.heatmap(confusion_matrix, 
                   xticklabels=[f'C{i}' for i in range(best_k)],
                   yticklabels=unique_emotions,
                   annot=True, 
                   fmt=viz_cfg.heatmap.fmt,
                   cmap=viz_cfg.heatmap.cmap,
                   ax=ax3,
                   cbar_kws={'label': 'Normalized Count'})
        ax3.set_title('Cluster-Class Relationship')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "clustering_analysis.png", 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. 메트릭 변화 시각화 (knee points 포함)
        self._plot_metrics(clustering_results["metrics"], 
                         clustering_results["knee_points"])
        
        # 수치 결과 저장
        self._save_numerical_results(clustering_results)

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

    def _plot_metrics(self, metrics: Dict[str, Dict[int, Dict[str, float]]], 
                      knee_points: Dict[str, float]):
        """메트릭 변화 시각화"""
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

    def _plot_metric_group(self, metrics: Dict[int, Dict[str, float]], 
                          title: str, filename: str, knee_point: float = None):
        """메트릭 그룹별 시각화"""
        plt.figure(figsize=(10, 6))
        
        # 데이터 정규화 및 플로팅
        for metric_name in metrics[list(metrics.keys())[0]].keys():
            values = [metrics[k][metric_name] for k in metrics.keys()]
            normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
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

    def _find_best_k_for_dec(self, dec: 'DeepEmbeddedClustering', data: np.ndarray, labels: List[str]) -> int:
        """DEC를 위한 최적의 k 찾기"""
        k_range = self.cfg.analysis.embedding.traditional.clustering.k_range
        scores = []
        
        # 잠재 특징 추출
        latent_features = dec.extract_features(data)
        
        print("\nFinding best k for DEC...")
        for k in tqdm(k_range, desc="Testing k values"):
            # k-means로 클러스터링
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
            cluster_labels = kmeans.fit_predict(latent_features)
            
            # 평가 메트릭 계산
            silhouette = silhouette_score(latent_features, cluster_labels)
            nmi = normalized_mutual_info_score(labels, cluster_labels)
            
            # 정규화된 점수의 평균
            combined_score = (silhouette + nmi) / 2
            scores.append(combined_score)
        
        # 최고 점수의 k 선택
        best_k_idx = np.argmax(scores)
        best_k = k_range[best_k_idx]
        
        print(f"\nBest k for DEC: {best_k} (Score: {scores[best_k_idx]:.4f})")
        return best_k

    def _evaluate_dec_results(self, cluster_labels: np.ndarray, true_labels: List[str], best_k: int) -> Dict[str, Any]:
        """DEC 결과 평가"""
        results = {
            "metrics": {
                "intrinsic": {},
                "extrinsic": {}
            },
            "best_k": best_k,
            "best_score": 0.0,
            "clustering_type": "dec"
        }
        
        # 메트릭 계산
        results["metrics"]["intrinsic"][best_k] = {
            "silhouette": silhouette_score(data, cluster_labels),
            "calinski_harabasz": calinski_harabasz_score(data, cluster_labels),
            "davies_bouldin_inverted": -davies_bouldin_score(data, cluster_labels)
        }
        
        results["metrics"]["extrinsic"][best_k] = {
            "adjusted_rand": adjusted_rand_score(true_labels, cluster_labels),
            "normalized_mutual_info": normalized_mutual_info_score(true_labels, cluster_labels)
        }
        
        # 최고 점수 저장
        results["best_score"] = results["metrics"]["extrinsic"][best_k]["normalized_mutual_info"]
        
        return results 