from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from openai import OpenAI
import json
import re
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv
model_name = "meta-llama-3.1-8b-instruct" #"emotion-llama@q8_0"
template_type = "arasu"

@dataclass
class EmotionResponse:
    emotion: str
    confidence_score: float

class PromptTemplate(Enum):
    DEFAULT = "default"
    SIMPLE = "simple"
    DETAILED = "detailed"
    ACADEMIC = "academic"
    ARASU = "arasu"

class BaseEmotionAnalyzer(ABC):
    def __init__(self, model_name: str, temperature: float = 0.3, template_type: str = "default"):
        self.model_name = model_name
        self.temperature = temperature
        self.labels = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']
        self.template_type = template_type
        self.prompt_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, str]:
        labels_str = "', '".join(self.labels)  # 동적으로 레이블 목록 생성
        return {
            PromptTemplate.DEFAULT.value: f"""Please analyze the emotional content of the following text and classify it as one of these emotions: '{labels_str}'.""",
            
            PromptTemplate.SIMPLE.value: """Classify the emotion as: joy, fear, anger, sadness, disgust, shame, or guilt.
Return JSON: {{"emotion": "emotion_name", "confidence_score": 0.0-1.0}}
Text: {text}""",
            
            PromptTemplate.DETAILED.value: """As an emotion analysis expert, please carefully analyze the following text and identify the primary emotion present. Consider the context, intensity, and subtle nuances in the language.

Available emotions for classification:
- Joy: positive feelings, happiness, pleasure
- Fear: threat, danger, anxiety
- Anger: frustration, rage, annoyance
- Sadness: loss, disappointment, grief
- Disgust: aversion, repulsion
- Shame: embarrassment, humiliation
- Guilt: responsibility for wrongdoing

Text to analyze: {text}

Provide your analysis in JSON format:
{{
    "emotion": "chosen_emotion",
    "confidence_score": "certainty_level_between_0_and_1"
}}""",
            
            PromptTemplate.ACADEMIC.value: """Conduct a systematic emotion classification analysis using the following framework:

1. Emotional Categories:
   - Joy (positive valence, high arousal)
   - Fear (negative valence, high arousal)
   - Anger (negative valence, high arousal)
   - Sadness (negative valence, low arousal)
   - Disgust (negative valence, medium arousal)
   - Shame (negative valence, low arousal)
   - Guilt (negative valence, variable arousal)

2. Analysis Parameters:
   - Linguistic markers
   - Emotional intensity
   - Contextual cues
   - Behavioral indicators

Text for analysis: {text}

Return results in JSON format:
{{
    "emotion": "classified_emotion",
    "confidence_score": "classification_confidence_0_to_1"
}}""",
            
            PromptTemplate.ARASU.value: f"""# Emotion Analysis Expert

You are an expert in emotion analysis. Your task is to classify the given text into EXACTLY ONE emotion from this list: '{labels_str}'.

CRITICAL INSTRUCTIONS:
1. You MUST return ONLY a JSON object with two fields:
   - "emotion": EXACTLY ONE of these: '{labels_str}'
   - "confidence_score": a number between 0 and 1

2. DO NOT add any other fields or explanations
3. DO NOT modify the emotion words
4. DO NOT use synonyms or similar words
5. Your response must be a valid JSON object

[Detailed emotion definitions and examples omitted for clarity]

Text to analyze: {{text}}

RESPONSE FORMAT (use exactly this structure):
{{
    "emotion": "one_of_the_specified_emotions",
    "confidence_score": 0.0-1.0
}}"""
        }
    
    def set_template(self, template_type: str):
        """Change the current template type"""
        if template_type in self.prompt_templates:
            self.template_type = template_type
        else:
            raise ValueError(f"Template type '{template_type}' not found. Available templates: {list(self.prompt_templates.keys())}")
    
    def add_custom_template(self, template_name: str, template_content: str):
        """Add a new custom template"""
        self.prompt_templates[template_name] = template_content
    
    def get_current_template(self) -> str:
        """Get the current template content"""
        return self.prompt_templates.get(self.template_type, self.prompt_templates[PromptTemplate.DEFAULT.value])
    
    def load_prompt_template(self, template_path: Optional[str] = None) -> str:
        if template_path:
            with open(template_path, 'r') as f:
                return f.read()
        return self.get_current_template()

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
    def __init__(self, model_name: str, base_url: str = "http://localhost:1234/v1", temperature: float = 0.3, template_type: str = "default"):
        super().__init__(model_name, temperature, template_type)
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
    
    def analyze(self, text: str) -> EmotionResponse:
        try:
            # 시스템 프롬프트 강화
            system_prompt = """You are an emotion analysis assistant. You MUST:
1. ALWAYS respond with a valid JSON object
2. ONLY use these emotions: joy, fear, anger, sadness, disgust, shame, guilt
3. NEVER refuse to analyze - for sensitive content, use 'sadness' with low confidence
4. Format: {"emotion": "one_of_allowed_emotions", "confidence_score": 0.0-1.0}"""

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.load_prompt_template().format(text=text)}
                ],
                temperature=self.temperature
            )
            
            content = completion.choices[0].message.content.strip()
            
            # 안전 필터 응답 처리
            safety_responses = [
                "i can't", "cannot", "don't", "unable to", 
                "not appropriate", "won't", "refused"
            ]
            if any(resp in content.lower() for resp in safety_responses):
                print(f"Safety filter triggered: {content}")
                return EmotionResponse(emotion="sadness", confidence_score=0.3)
            
            try:
                # 단순화된 JSON 추출
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if not json_match:
                    raise ValueError(f"No valid JSON found in: {content}")
                
                json_str = json_match.group().strip()
                valid_json = json.loads(json_str)
                
                if not ("emotion" in valid_json and "confidence_score" in valid_json):
                    raise ValueError("Invalid JSON structure")
                
                # 감정 정규화 및 매핑
                emotion = str(valid_json["emotion"]).lower().strip()
                emotion = self._normalize_emotion(emotion)
                confidence = float(valid_json["confidence_score"])
                
                return EmotionResponse(
                    emotion=emotion,
                    confidence_score=min(max(confidence, 0.0), 1.0)
                )
                
            except Exception as e:
                print(f"Error parsing response: {e}\nContent: {content}")
                return EmotionResponse(emotion="sadness", confidence_score=0.3)
                
        except Exception as e:
            print(f"Error in LM Studio analysis: {e}")
            return EmotionResponse(emotion="sadness", confidence_score=0.3)

    def _normalize_emotion(self, emotion: str) -> str:
        """감정 정규화 및 매핑"""
        emotion_mapping = {
            'neutral': 'sadness',
            'anxiety': 'fear',
            'gratitude': 'joy',
            'happy': 'joy',
            'scared': 'fear',
            'angry': 'anger',
            'frustrated': 'anger',
            'disgusted': 'disgust',
            'ashamed': 'shame',
            'guilty': 'guilt',
            'surprised': 'fear',
            'worried': 'fear',
            'confused': 'fear',
            'hurt': 'sadness',
            'disappointed': 'sadness'
        }
        
        # 먼저 직접 매핑 시도
        emotion = emotion_mapping.get(emotion, emotion)
        
        # 유효한 감정이 아니면 가장 가까운 것 찾기
        if emotion not in self.labels:
            print(f"Warning: Invalid emotion '{emotion}', finding closest match")
            emotion = self._find_closest_emotion(emotion)
        
        return emotion

    def _find_closest_emotion(self, invalid_emotion: str) -> str:
        """가장 유사한 감정 찾기"""
        from difflib import get_close_matches
        
        # 먼저 매핑 테이블에서 확인
        emotion_groups = {
            'joy': ['happy', 'pleased', 'delighted', 'grateful', 'satisfied'],
            'fear': ['scared', 'anxious', 'worried', 'nervous', 'terrified'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'frustrated'],
            'sadness': ['sad', 'unhappy', 'depressed', 'disappointed', 'hurt'],
            'disgust': ['disgusted', 'repulsed', 'revolted'],
            'shame': ['ashamed', 'embarrassed', 'humiliated'],
            'guilt': ['guilty', 'remorseful', 'regretful']
        }
        
        # 그룹에서 매칭 확인
        for emotion, synonyms in emotion_groups.items():
            if invalid_emotion in synonyms:
                return emotion
        
        # 유사도 기반 매칭
        matches = get_close_matches(invalid_emotion, self.labels, n=1, cutoff=0.6)
        return matches[0] if matches else "sadness"

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



def test_lmstudio_connection(model_name: str):
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
            model=model_name,  # 실제 로드된 모델명으로 변경
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
    analyzer = LMStudioEmotionAnalyzer(
        model_name=model_name,#"llama2",
        template_type=template_type #"default"
    )
    analyzer.set_template("detailed")
    # 커스텀 템플릿 추가
    analyzer.add_custom_template("my_template", """
    Analyze the emotion in this text: {text}
    Return: {{"emotion": "emotion_type", "confidence_score": 0.0-1.0}}
    """)

    # 커스텀 템플릿 사용
    analyzer.set_template("my_template")
    # 데이터 처리
    df = pd.read_csv('./arasu/isear_dataset_new.csv')
    output_path = Path(f'./arasu/isear_dataset_new.csv')

    # # OpenAI로 처리
    # processor = EmotionAnalysisProcessor(openai_analyzer, output_path)
    # df = processor.process_dataset(df, 'text')
    print("Testing LM Studio connection...")
    is_connected = test_lmstudio_connection(model_name)
    print(f"Connection status: {'Success' if is_connected else 'Failed'}")
    # 템플릿별 결과 비교
    templates_to_test = ["default", "simple", "detailed", "academic", "my_template"]
    results = {}
    for template in templates_to_test:
        analyzer.set_template(template)
        result = analyzer.analyze("I am very happy today!")
        results[template] = result

    # 결과 비교
    for template, result in results.items():
        print(f"\nTemplate: {template}")
        print(f"Emotion: {result.emotion}")
        print(f"Confidence: {result.confidence_score}")
    #LM Studio로 처리
    processor = EmotionAnalysisProcessor(analyzer, output_path)
    df = processor.process_dataset(df, 'text')

