from .base import BaseGenerator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from typing import List, Dict
import openai
import requests
import json

class Generator(BaseGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.initialize()
        
    def initialize(self):
        """LLM과 프롬프트 초기화"""
        self.model = self._initialize_model()
        self.prompt = self._create_prompt()
        
        if self.cfg.get('debug', {}).get('show_prompt', False):
            print("\nPrompt Template:")
            print("-" * 80)
            print(self.prompt.messages[0].prompt)
            print("-" * 80)
            
        self.provider = self.cfg.model.provider
        self.emotion_map = {str(i+1): emotion for i, emotion in enumerate(self.cfg.emotions.classes)}
        
        if self.provider == "openai":
            self.model = ChatOpenAI(
                model_name=self.cfg.model.openai.chat_model_name,
                temperature=0
            )
        elif self.provider == "lmstudio":
            self.base_url = self.cfg.model.lmstudio.base_url
            self.client = requests.Session()
        
    def _initialize_model(self):
        if self.cfg.model.provider == "openai":
            return ChatOpenAI(
                model_name=self.cfg.model.openai.chat_model_name
            )
        else:
            return ChatOpenAI(
                base_url=self.cfg.model.lmstudio.base_url,
                api_key=self.cfg.model.lmstudio.api_key
            )
            
    def _create_prompt(self):
        """프롬프트 템플릿 생성"""
        if self.cfg.model.use_rag:
            template = self.cfg.model.templates.rag
        else:
            template = self.cfg.model.templates.base
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_examples(self, contexts: List[Dict]) -> str:
        """검색된 문맥을 구조화된 few-shot 예시로 변환"""
        formatted_examples = []
        
        for ctx in contexts:
            # 감정 레이블 변환 (인덱스 -> 감정 이름)
            emotion_label = ctx['metadata']['emotion']
            if emotion_label.isdigit() and emotion_label in self.emotion_map:
                emotion_name = self.emotion_map[emotion_label]
            else:
                emotion_name = emotion_label.lower()
            
            # 유사도 점수 포함
            similarity_score = ctx.get('score', 0.0)
            example = f"""Text: {ctx['page_content']}
Emotion: {emotion_name}
(Similarity: {similarity_score:.4f})"""
            formatted_examples.append(example)
            
        return "\n\n".join(formatted_examples)

    def generate(self, contexts: List[Dict], query: str) -> str:
        """감정 분석 수행"""
        if self.cfg.model.use_rag:
            examples = self._format_examples(contexts)
            prompt_args = {
                "emotions": ", ".join(self.cfg.emotions.classes),
                "examples": examples,
                "input": query
            }
        else:
            prompt_args = {
                "emotions": ", ".join(self.cfg.emotions.classes),
                "input": query
            }
        
        # 전체 프롬프트 생성
        full_prompt = self.prompt.format(**prompt_args)
        
        # 디버그 출력
        if self.cfg.get('debug', {}).get('show_generation', False):
            print("\nGeneration Input:")
            print("-" * 80)
            if self.cfg.get('debug', {}).get('show_full_prompt', False):
                print("Full Prompt:")
                print(full_prompt)
                print("-" * 80)
            print("\nRAG:", "Enabled" if self.cfg.model.use_rag else "Disabled")
            print("\nAllowed Emotions:", self.cfg.emotions.classes)
            print("\nQuery:", query)
            print("-" * 80)

        # 프롬프트 실행 및 응답 처리
        if self.provider == "openai":
            response = self.model.predict(full_prompt)
            raw_response = response.strip().lower()
            print(f"\nRaw Response: '{raw_response}'")  # 디버그용
        elif self.provider == "lmstudio":
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Generation failed: {response.json()}")
                
            raw_response = response.json()["choices"][0]["message"]["content"].strip().lower()
            print(f"\nRaw Response: '{raw_response}'")  # 디버그용

        # 응답 정규화 및 검증
        pred_emotion = raw_response.split()[-1] if raw_response else ""  # 마지막 단어만 사용
        
        # 예측된 감정이 허용된 클래스에 있는지 확인
        if pred_emotion not in self.cfg.emotions.classes:
            print(f"\nWarning: Predicted emotion '{pred_emotion}' not in allowed classes.")
            # 가장 유사한 감정으로 매핑
            closest_emotion = min(
                self.cfg.emotions.classes,
                key=lambda x: len(set(x) - set(pred_emotion))  # 간단한 문자열 유사도
            )
            print(f"Mapping to closest emotion: '{closest_emotion}'")
            pred_emotion = closest_emotion

        print(f"Final Prediction: '{pred_emotion}'")  # 디버그용
        print("-" * 80)
        
        return pred_emotion 