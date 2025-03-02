from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pandas as pd
from pathlib import Path
import numpy as np
import pickle

class EmotionRAG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db_path = Path(cfg.rag.db_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.rag.embedding_model)
        self.vectorstore = None
        self.exclude_indices = set()
        
        # 감정 분석 특화 RAG 템플릿 정의
        self.emotion_template = """You are an expert in emotion analysis. Your task is to analyze the emotional content of the given text based on similar examples and your knowledge.

## Similar Examples for Reference:
{examples}

## Current Text to Analyze:
{query}

## Instructions:
1. Consider the context from similar examples above
2. Analyze the emotional content of the current text
3. Classify the emotion STRICTLY as one of: {labels}
4. NO OTHER EMOTIONS are allowed

## Guidelines:
- Focus on the dominant emotion
- Use similar examples as reference points
- Consider emotional intensity and context
- Be consistent with example classifications

## Response Format:
Provide your analysis in JSON format with these fields:
- emotion: (must be one of the specified labels)
- confidence_score: (between 0.0 and 1.0)
- explanation: (brief reasoning based on similar examples and context)

Return ONLY the JSON response without any additional text or formatting."""
        
        self.prompt_template = PromptTemplate(
            input_variables=["examples", "query", "labels"],
            template=self.emotion_template
        )
    
    def create_index(self, df):
        """벡터 DB 생성 또는 로드"""
        db_file = self.db_path / f"{self.cfg.data.name}_vectorstore.pkl"
        
        # 기존 DB 로드 시도
        if self.cfg.rag.load_db and db_file.exists():
            print(f"Loading existing vector DB from {db_file}")
            try:
                with open(db_file, 'rb') as f:
                    self.vectorstore = pickle.load(f)
                return
            except Exception as e:
                print(f"Error loading vector DB: {e}")
        
        # 새로운 DB 생성
        print("Creating new vector DB...")
        texts = df['text'].tolist()
        metadatas = [
            {
                'emotion': row.emotion,
                'index': idx
            } for idx, row in df.iterrows()
        ]
        
        self.vectorstore = FAISS.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas
        )
        
        # DB 저장
        if self.cfg.rag.save_db:
            print(f"Saving vector DB to {db_file}")
            self.db_path.mkdir(parents=True, exist_ok=True)
            with open(db_file, 'wb') as f:
                pickle.dump(self.vectorstore, f)
    
    def get_similar_examples(self, query: str, k: int = 3) -> list:
        """유사한 예제 검색"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_index first.")
            
        # 현재 인덱스 제외하고 검색
        search_kwargs = {}
        if self.exclude_indices:
            search_kwargs["filter"] = {"index": {"$nin": list(self.exclude_indices)}}
            
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=k, **search_kwargs
        )
        return docs_and_scores
    
    def get_rag_prompt(self, query: str, similar_examples: list) -> str:
        """RAG 프롬프트 생성"""
        # 유사 예제 포맷팅
        formatted_examples = []
        for doc, score in similar_examples:
            formatted_examples.append(
                f"Text: {doc.page_content}\n"
                f"Emotion: {doc.metadata['emotion']}\n"
                f"Similarity: {score:.3f}"
            )
        examples_text = "\n\n".join(formatted_examples)
        
        # 라벨 목록 가져오기
        labels = ", ".join(self.cfg.data.datasets[self.cfg.data.name].labels)
        
        # 프롬프트 생성
        prompt = self.prompt_template.format(
            examples=examples_text,
            query=query,
            labels=labels
        )
        
        return prompt
    
    def exclude_index(self, idx: int):
        """특정 인덱스 제외 목록에 추가"""
        self.exclude_indices.add(idx) 