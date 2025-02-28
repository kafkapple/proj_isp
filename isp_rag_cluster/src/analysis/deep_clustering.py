import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Dict
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, Model

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
    def __init__(self, input_dim, n_clusters, encoder_dims=[500, 500, 2000, 10],
                 alpha=1.0, pretrain_epochs=100, clustering_epochs=100,
                 batch_size=256, update_interval=140, tol=0.001, learning_rate=0.001):
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.encoder_dims = encoder_dims
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.tol = tol
        self.learning_rate = learning_rate
        
        # Build models
        self.autoencoder, self.encoder = self._build_autoencoder()
        self.clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=self.clustering_layer)

    def _build_autoencoder(self):
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = input_layer
        
        for dim in self.encoder_dims[:-1]:
            encoded = layers.Dense(dim, activation='relu')(encoded)
        
        # Final encoder layer
        encoded = layers.Dense(self.encoder_dims[-1], name='encoder')(encoded)
        
        # Create encoder model
        encoder = Model(inputs=input_layer, outputs=encoded, name='encoder')
        
        # Decoder
        decoded = encoded
        for dim in reversed(self.encoder_dims[:-1]):
            decoded = layers.Dense(dim, activation='relu')(decoded)
        
        decoded = layers.Dense(self.input_dim, name='decoder')(decoded)
        
        # Create autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
        
        return autoencoder, encoder

    def fit_predict(self, X):
        # Pretrain autoencoder
        print("Pretraining autoencoder...")
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                               loss='mse')
        self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=self.pretrain_epochs,
                           verbose=1)

        # Initialize cluster centers using k-means
        print("\nInitializing cluster centers...")
        features = self.encoder.predict(X)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(features)
        self.model.get_layer('clustering').set_weights([kmeans.cluster_centers_])

        # Deep clustering
        print("\nDeep clustering...")
        
        # KL divergence loss 함수 정의
        def kl_divergence_loss(y_true, y_pred):
            return tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_true / (y_pred + 1e-6)), axis=-1))
        
        # 모델 컴파일 with KL divergence loss
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=kl_divergence_loss
        )
        
        # Initialize predictions
        y_pred_last = kmeans.labels_
        y_pred = y_pred_last.copy()
        
        # Train the model
        for epoch in range(self.clustering_epochs):
            if epoch % self.update_interval == 0:
                q = self.model.predict(X)
                p = self._target_distribution(q)  # target distribution P
                y_pred = q.argmax(1)
                
                # Check convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred.copy()
                
                if epoch > 0 and delta_label < self.tol:
                    print('Reached tolerance threshold. Stopping training.')
                    break
            
            # Train on batch with target distribution
            self.model.train_on_batch(X, p)
            
        return y_pred

    def _target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

class ClusteringLayer(layers.Layer):
    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = None

    def build(self, input_shape):
        self.clusters = self.add_weight(shape=(self.n_clusters, input_shape[-1]),
                                      initializer='glorot_uniform',
                                      name='clusters')

    def call(self, inputs):
        q = 1.0 / (1.0 + tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), 
            axis=2
        ) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        return q / tf.reduce_sum(q, axis=1, keepdims=True)

    def get_config(self):
        config = super().get_config()
        config.update({'n_clusters': self.n_clusters, 'alpha': self.alpha})
        return config

    def pretrain(self, data: np.ndarray):
        """오토인코더 사전 학습"""
        dataset = TensorDataset(torch.FloatTensor(data))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        
        pbar = tqdm(range(self.pretrain_epochs), desc="Pretraining")
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
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        
        # 클러스터링 학습
        pbar = tqdm(range(self.clustering_epochs), desc="Clustering")
        for epoch in pbar:
            if epoch % self.update_interval == 0:
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