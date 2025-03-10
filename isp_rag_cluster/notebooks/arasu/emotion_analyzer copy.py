from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from openai import OpenAI
import json
import re
from pathlib import Path
import pandas as pd
from dataclasses import dataclass

@dataclass
class EmotionResponse:
    emotion: str
    confidence_score: float

class BaseEmotionAnalyzer(ABC):
    def __init__(self, model_name: str, temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self.labels = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']
    
    @abstractmethod
    def analyze(self, text: str) -> EmotionResponse:
        pass
    
    def load_prompt_template(self, template_path: Optional[str] = None) -> str:
        if template_path:
            with open(template_path, 'r') as f:
                return f.read()
        return self.default_prompt_template()
    
    def default_prompt_template(self) -> str:
        return """Please analyze the emotional content of the following text and classify it as one of these emotions: joy, fear, anger, sadness, disgust, shame, or guilt.

Respond in JSON format with two fields:
1. "emotion": one of the specified emotions
2. "confidence_score": a number between 0 and 1

Text to analyze: {text}

Response format:
{{
    "emotion": "emotion_name",
    "confidence_score": 0.0-1.0
}}"""

class OpenAIEmotionAnalyzer(BaseEmotionAnalyzer):
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.3):
        super().__init__(model_name, temperature)
        self.client = OpenAI(api_key=api_key)
    
    def analyze(self, text: str) -> EmotionResponse:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": self.load_prompt_template().format(text=text)}],
                temperature=self.temperature
            )
            content = completion.choices[0].message.content
            result = json.loads(content)
            return EmotionResponse(
                emotion=result["emotion"].lower(),
                confidence_score=float(result["confidence_score"])
            )
        except Exception as e:
            print(f"Error in OpenAI analysis: {e}")
            return EmotionResponse(emotion="unknown", confidence_score=0.0)

class LMStudioEmotionAnalyzer(BaseEmotionAnalyzer):
    def __init__(self, model_name: str, base_url: str = "http://localhost:1234/v1", temperature: float = 0.3):
        super().__init__(model_name, temperature)
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
    
    def analyze(self, text: str) -> EmotionResponse:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an emotion analysis assistant. Always respond with a valid JSON object containing 'emotion' and 'confidence_score' fields."},
                    {"role": "user", "content": self.load_prompt_template().format(text=text)}
                ],
                temperature=self.temperature
            )
            
            content = completion.choices[0].message.content.strip()
            try:
                # JSON 문자열에서 실제 JSON 객체 찾기
                json_str = re.search(r'\{.*\}', content, re.DOTALL)
                if json_str:
                    result = json.loads(json_str.group())
                    if "emotion" in result and "confidence_score" in result:
                        return EmotionResponse(
                            emotion=str(result["emotion"]).lower(),
                            confidence_score=float(result["confidence_score"])
                        )
                print(f"Invalid response format: {content}")
                return EmotionResponse(emotion="unknown", confidence_score=0.0)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing response: {e}\nContent: {content}")
                return EmotionResponse(emotion="unknown", confidence_score=0.0)
                
        except Exception as e:
            print(f"Error in LM Studio analysis: {e}")
            return EmotionResponse(emotion="unknown", confidence_score=0.0)

class EmotionAnalysisProcessor:
    def __init__(self, analyzer: BaseEmotionAnalyzer, output_path: Path):
        self.analyzer = analyzer
        self.output_path = output_path
    
    def process_dataset(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        # 필요한 컬럼 미리 생성
        emotion_col = f"predicted_emotion_{self.analyzer.model_name}"
        confidence_col = f"confidence_score_{self.analyzer.model_name}"
        
        # 컬럼이 없으면 생성
        if emotion_col not in df.columns:
            df[emotion_col] = "unknown"
        if confidence_col not in df.columns:
            df[confidence_col] = 0.0
        
        for index, row in df.iterrows():
            if row[emotion_col] not in self.analyzer.labels:
                response = self.analyzer.analyze(row[column_name])
                df.at[index, emotion_col] = response.emotion
                df.at[index, confidence_col] = response.confidence_score
                
                print(f"Row {index}| Ground truth - {row['emotion']} and predicted - {response.emotion}")
                
                if index % 500 == 0:
                    df.to_csv(self.output_path, index=False)
        
        df.to_csv(self.output_path, index=False)
        return df

import os
from dotenv import load_dotenv
model_name = "meta-llama-3.1-8b-instruct"

def test_lmstudio_connection():
    try:
        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )
        
        # 모델 목록 확인
        models = client.models.list()
        print("Available models:", models)
        
        # 간단한 테스트 요청
        completion = client.chat.completions.create(
            model="meta-llama-3.1-8b-instruct",  # 실제 로드된 모델명으로 변경
            messages=[
                {"role": "user", "content": "Hello, are you there?"}
            ]
        )
        print("Test response:", completion.choices[0].message.content)
        
        return True
    except Exception as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    # Initialize LM Studio client
    # OpenAI 분석기 설정
    # openai_analyzer = OpenAIEmotionAnalyzer(
    #     model_name="gpt-3.5-turbo",
    #     api_key=api_key
    # )

    # LM Studio 분석기 설정
    lmstudio_analyzer = LMStudioEmotionAnalyzer(
        model_name=model_name,#"llama2",
        base_url="http://localhost:1234/v1"
    )

    # 데이터 처리
    df = pd.read_csv('./arasu/isear_dataset.csv')
    output_path = Path('./arasu/isear_dataset_new.csv')

    # # OpenAI로 처리
    # processor = EmotionAnalysisProcessor(openai_analyzer, output_path)
    # df = processor.process_dataset(df, 'text')
    print("Testing LM Studio connection...")
    is_connected = test_lmstudio_connection()
    print(f"Connection status: {'Success' if is_connected else 'Failed'}")

    # LM Studio로 처리
    processor = EmotionAnalysisProcessor(lmstudio_analyzer, output_path)
    df = processor.process_dataset(df, 'text')

