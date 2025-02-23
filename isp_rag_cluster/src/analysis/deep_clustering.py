import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Dict
from tqdm import tqdm

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        
        # 인코더
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims[:-1]:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 잠재 표현
        self.latent = nn.Linear(prev_dim, hidden_dims[-1])
        
        # 디코더
        decoder_layers = []
        prev_dim = hidden_dims[-1]
        for dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        latent = self.latent(encoded)
        decoded = self.decoder(latent)
        return latent, decoded

class DeepEmbeddedClustering:
    def __init__(self, cfg, input_dim: int):
        self.cfg = cfg.analysis.embedding.dec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 초기화
        self.autoencoder = AutoEncoder(
            input_dim=input_dim,
            hidden_dims=self.cfg.hidden_dims
        ).to(self.device)
        
        # 클러스터 중심점
        self.cluster_centers = None
        self.alpha = 1.0  # student's t-distribution 자유도
        
    def pretrain(self, data: np.ndarray):
        """오토인코더 사전 학습"""
        dataset = TensorDataset(torch.FloatTensor(data))
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.cfg.learning_rate)
        
        pbar = tqdm(range(self.cfg.pretrain_epochs), desc="Pretraining")
        for epoch in pbar:
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                _, decoded = self.autoencoder(x)
                loss = nn.MSELoss()(decoded, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss/len(dataloader)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
    
    def initialize_cluster_centers(self, data: np.ndarray, n_clusters: int):
        """k-means로 초기 클러스터 중심점 설정"""
        latent_features = self.extract_features(data)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        self.cluster_centers = torch.tensor(
            kmeans.fit_predict(latent_features),
            dtype=torch.float,
            device=self.device
        )
    
    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """목표 분포 계산"""
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
    
    def cluster(self, data: np.ndarray, n_clusters: int) -> Dict:
        """DEC 클러스터링 수행"""
        print("\nInitializing cluster centers...")
        self.initialize_cluster_centers(data, n_clusters)
        
        dataset = TensorDataset(torch.FloatTensor(data))
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.cfg.learning_rate)
        
        # 클러스터링 학습
        pbar = tqdm(range(self.cfg.clustering_epochs), desc="Clustering")
        for epoch in pbar:
            if epoch % self.cfg.update_interval == 0:
                # 클러스터 할당 확률 계산
                q = self._compute_soft_assignments(data)
                p = self.target_distribution(q)
            
            total_loss = 0
            total_kl_loss = 0
            total_rec_loss = 0
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                
                # 클러스터링 손실
                z, _ = self.autoencoder(x)
                q_batch = self._compute_soft_assignments(z)
                p_batch = self.target_distribution(q_batch)
                kl_loss = nn.KLDivLoss(reduction='batchmean')(q_batch.log(), p_batch)
                
                # 재구성 손실
                _, decoded = self.autoencoder(x)
                rec_loss = nn.MSELoss()(decoded, x)
                
                # 전체 손실
                loss = kl_loss + rec_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_kl_loss += kl_loss.item()
                total_rec_loss += rec_loss.item()
            
            # 평균 손실 계산
            avg_loss = total_loss/len(dataloader)
            avg_kl_loss = total_kl_loss/len(dataloader)
            avg_rec_loss = total_rec_loss/len(dataloader)
            
            # 진행 상황 업데이트
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "KL": f"{avg_kl_loss:.4f}",
                "Rec": f"{avg_rec_loss:.4f}"
            })
        
        # 최종 클러스터 할당
        final_assignments = self._compute_soft_assignments(data).argmax(1).cpu().numpy()
        latent_features = self.extract_features(data)
        
        return {
            "cluster_labels": final_assignments,
            "latent_features": latent_features
        }
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """잠재 특징 추출"""
        self.autoencoder.eval()
        with torch.no_grad():
            latent, _ = self.autoencoder(torch.FloatTensor(data).to(self.device))
        return latent.cpu().numpy() 