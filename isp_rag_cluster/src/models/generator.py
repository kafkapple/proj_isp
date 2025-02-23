from .base import BaseGenerator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from typing import List, Dict, Optional
import openai
import requests
import json
from difflib import SequenceMatcher

class Generator(BaseGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.initialize()
        
    def initialize(self):
        """Initialize LLM and prompt"""
        self.provider = self.cfg.model.provider
        self.prompt = self._create_prompt()
        
        if self.cfg.get('debug', {}).get('show_prompt', False):
            print("\nPrompt Template:")
            print("-" * 80)
            print(self.prompt.messages[0].prompt)
            print("-" * 80)
            
        self.emotion_map = {str(i+1): emotion for i, emotion in enumerate(self.cfg.emotions.classes)}
        
        if self.provider == "openai":
            self.model = ChatOpenAI(
                model_name=self.cfg.model.openai.chat_model_name,
                temperature=0
            )
        elif self.provider == "lmstudio":
            self.base_url = self.cfg.model.lmstudio.base_url
            self.client = requests.Session()
            self.model_info = self._get_model_info()
    
    def _get_model_info(self) -> dict:
        """Get current loaded model information from LMStudio"""
        try:
            response = self.client.get(f"{self.base_url}/models")
            if response.status_code == 200:
                models = response.json()
                if models and "data" in models and models["data"]:
                    model = models["data"][0]
                    return {
                        "id": model.get("id", "unknown"),
                        "created": model.get("created", "unknown"),
                        "object": model.get("object", "unknown"),
                        "owned_by": model.get("owned_by", "unknown")
                    }
            return {"id": "unknown", "error": "Failed to get model info"}
        except Exception as e:
            print(f"Warning: Failed to fetch model info: {e}")
            return {"id": "unknown", "error": str(e)}

    def get_model_info(self) -> dict:
        """Return current model information"""
        if self.provider == "openai":
            return {
                "id": self.cfg.model.openai.chat_model_name,
                "provider": "openai"
            }
        elif self.provider == "lmstudio":
            return {
                "provider": "lmstudio",
                "base_url": self.base_url,
                "model": getattr(self, 'model_info', {"id": "unknown"})
            }
        return {"id": "unknown", "provider": self.provider}

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
        """Create prompt template"""
        if self.cfg.model.use_rag:
            template = self.cfg.model.templates.rag
        else:
            template = self.cfg.model.templates.base
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Debug output
        if self.cfg.debug.show_prompt:
            print("\nPrompt Template:")
            print("-" * 80)
            print(template)  # 템플릿 자체 출력
            print("-" * 80)
        
        return prompt
    
    def _format_examples(self, contexts: List[Dict]) -> str:
        """Convert searched contexts into structured few-shot examples"""
        formatted_examples = []
        
        for ctx in contexts:
            # Convert emotion label (index -> emotion name)
            emotion_label = ctx['metadata']['emotion']
            if emotion_label.isdigit() and emotion_label in self.emotion_map:
                emotion_name = self.emotion_map[emotion_label]
            else:
                emotion_name = emotion_label.lower()
            
            # Include similarity score
            similarity_score = ctx.get('score', 0.0)
            example = f"""Text: {ctx['page_content']}
Emotion: {emotion_name}
(Similarity: {similarity_score:.4f})"""
            formatted_examples.append(example)
            
        return "\n\n".join(formatted_examples)

    def _format_prompt(self, text: str) -> str:
        """Format base prompt without context"""
        prompt_args = {
            "input": text,
            "emotions": ", ".join(self.cfg.emotions.classes)  # emotion classes 리스트를 문자열로 변환
        }
        return self.prompt.format(**prompt_args)

    def _format_prompt_with_context(self, context: str, text: str) -> str:
        """Format prompt with RAG context"""
        prompt_args = {
            "input": text,
            "emotions": ", ".join(self.cfg.emotions.classes),  # emotion classes 리스트를 문자열로 변환
            "examples": self._format_examples(context) if context else ""
        }
        return self.prompt.format(**prompt_args)

    def generate(self, context: Optional[str], text: str) -> str:
        """Generate emotion prediction"""
        if self.cfg.model.use_rag and context:
            prompt = self._format_prompt_with_context(context, text)
        else:
            prompt = self._format_prompt(text)
        
        # Debug: Show formatted prompt
        if self.cfg.debug.show_full_prompt:
            print("\nFormatted Prompt:")
            print("-" * 80)
            print(prompt)
            print("-" * 80)
        
        # Execute prompt and process response
        if self.provider == "openai":
            response = self.model.predict(prompt)
            raw_response = response.strip().lower()
        elif self.provider == "lmstudio":
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Generation failed: {response.json()}")
            
            raw_response = response.json()["choices"][0]["message"]["content"].strip().lower()
        
        # Normalize response and validate
        raw_response = raw_response.strip().lower()
        
        # Extract last word and clean it
        words = [w for w in raw_response.split() if w.strip()]
        pred_emotion = words[-1] if words else ""
        
        # Debug output
        if self.cfg.debug.show_generation:
            print(f"\nRaw Response: '{raw_response}'")
            print(f"Extracted emotion: '{pred_emotion}'")
        
        # Check if predicted emotion is in allowed classes
        if pred_emotion not in self.cfg.emotions.classes:
            print(f"\nWarning: Predicted emotion '{pred_emotion}' not in allowed classes.")
            
            # 1. 문자열 유사도 기반 매핑
            def similarity_ratio(a, b):
                return SequenceMatcher(None, a, b).ratio()
            
            # 각 허용된 감정에 대한 유사도 계산
            similarities = {
                emotion: similarity_ratio(pred_emotion, emotion)
                for emotion in self.cfg.emotions.classes
            }
            
            # 가장 유사한 감정 선택
            closest_emotion = max(similarities.items(), key=lambda x: x[1])
            
            print(f"Mapping '{pred_emotion}' to '{closest_emotion[0]}' (similarity: {closest_emotion[1]:.2f})")
            pred_emotion = closest_emotion[0]
        
        if self.cfg.debug.show_generation:
            print(f"Final Prediction: '{pred_emotion}'")
            print("-" * 80)
        
        return pred_emotion

    def _print_debug_info(self):
        # Implementation of _print_debug_info method
        pass 