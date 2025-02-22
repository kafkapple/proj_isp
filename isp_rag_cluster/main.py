import os
from dotenv import load_dotenv

from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import requests
from langchain_core.embeddings import Embeddings
from typing import List

from src.data.data_manager import DataManager
from src.models.retriever import Retriever
from src.models.generator import Generator
from src.evaluation.evaluator import Evaluator

# .env 파일에서 환경 변수 로드
load_dotenv()

class LMStudioEmbeddings(Embeddings):
    def __init__(self, base_url: str, api_key: str = "not-needed"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = requests.Session()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """임베딩 생성"""
        all_embeddings = []
        
        for text in tqdm(texts, desc="Generating embeddings"):
            response = self.client.post(
                f"{self.base_url}/embeddings",
                headers={"Content-Type": "application/json"},
                json={
                    "input": text,
                    "model": "embedding"
                }
            )
            if response.status_code != 200:
                raise ValueError(f"Embedding failed: {response.json()}")
            embedding = response.json()["data"][0]["embedding"]
            all_embeddings.append(embedding)
            
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩 생성"""
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers={"Content-Type": "application/json"},
            json={
                "input": text,
                "model": "embedding"
            }
        )
        if response.status_code != 200:
            raise ValueError(f"Query embedding failed: {response.json()}")
        return response.json()["data"][0]["embedding"]

class EmotionAnalyzer:
    def __init__(self, train_df, cfg):
        self.retriever = Retriever(train_df, cfg)
        self.generator = Generator(cfg)
        self.embedding_info = self.retriever.embeddings.embedding_info

    def analyze_emotion(self, text: str) -> str:
        retrieved_context = self.retriever.retrieve(text)
        return self.generator.generate(retrieved_context, text)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # 데이터 로드 및 분할
    data_manager = DataManager(cfg)
    train_df, val_df = data_manager.split_data()
    
    # 감정 분석기 초기화
    analyzer = EmotionAnalyzer(train_df, cfg)
    
    # 평가 (validation set만 사용)
    evaluator = Evaluator(analyzer, cfg)
    metrics = evaluator.evaluate(val_df)
    
    print("\nValidation Results:")
    for metric_name, value in metrics["overall"].items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()