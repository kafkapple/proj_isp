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
    def __init__(self, cfg, model_info: dict = None):
        super().__init__(cfg)
        self.model_info = model_info
        self.initialize()
        
    def initialize(self):
        """Initialize LLM and prompt"""
        self.provider = self.cfg.model.provider
        self.prompt = self._create_prompt()
        
        self.emotion_map = {str(i+1): emotion for i, emotion in enumerate(self.cfg.emotions.classes)}
        
        # Initialize the model
        if self.provider == "openai":
            self.model = ChatOpenAI(
                model_name=self.cfg.model.openai.chat_model_name,
                temperature=self.cfg.model.openai.temperature
            )
        elif self.provider == "lmstudio":
            self.base_url = self.cfg.model.lmstudio.base_url
            self.client = requests.Session()
            self.model = ChatOpenAI(
                base_url=self.base_url,
                api_key=self.cfg.model.lmstudio.api_key,
                model=self.model_info.get("id") if self.model_info else None,
                temperature=self.cfg.model.lmstudio.temperature
            )
    
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
        
        if self.cfg.debug.show_prompt:
            print("\nPrompt Template:")
            print("-" * 80)
            print(template)
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
        """Format prompt without context"""
        return self.prompt.format(input=text)

    def _format_prompt_with_context(self, context: str, text: str) -> str:
        """Format prompt with context"""
        return self.prompt.format(
            context=context,
            input=text
        )

    def generate(self, context: Optional[List[Dict]], text: str) -> str:
        """Generate response using the appropriate prompt template"""
        if context:
            # RAG case: format prompt with context
            if self.cfg.debug.show_retrieval:
                print("\nRetrieved Context:")
                print("-" * 80)
                print(context)
                print("-" * 80)
            
            prompt = self._format_prompt_with_context(context, text)
        else:
            # Base case: format prompt without context
            prompt = self._format_prompt(text)

        # Show full prompt if debug enabled
        if self.cfg.debug.show_full_prompt:
            print("\nFull Prompt:")
            print("-" * 80)
            print(prompt)
            print("-" * 80)

        # Generate response
        response = self.model.invoke(prompt)
        result = self._parse_response(response.content)
        
        # 유효성 검사
        valid_emotions = self.cfg.emotions.classes
        if result not in valid_emotions:
            print(f"Warning: Invalid emotion '{result}' returned by model")
            # 기본값으로 가장 적절한 감정 반환 또는 에러 처리
            return "sadness"  # 또는 다른 처리 방법 선택
        
        return result

    def _print_debug_info(self):
        # Implementation of _print_debug_info method
        pass 

    def _parse_response(self, response: str) -> str:
        """Parse model response to extract emotion"""
        try:
            # Remove any leading/trailing whitespace and convert to lowercase
            response = response.strip().lower()
            
            # Try to parse as JSON
            import json
            response_json = json.loads(response)
            if isinstance(response_json, dict):
                # Extract emotion from JSON response
                if "emotion" in response_json:
                    emotion = response_json["emotion"].lower()
                    return emotion
        except json.JSONDecodeError:
            # If not JSON, return the cleaned response
            return response
        except Exception as e:
            print(f"Warning: Error parsing response: {e}")
            return response
        
        # If we get here, return the original response
        return response 