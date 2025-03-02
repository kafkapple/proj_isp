from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import json
import re
from ollama import chat, ChatResponse
from anthropic import Anthropic  # Anthropic 패키지 추가
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
load_dotenv()
from pathlib import Path
import hydra
from tqdm import tqdm
from openai import OpenAI
from omegaconf import OmegaConf
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from rag import EmotionRAG
import logging
import sys
from typing import Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings  # 새로운 임포트 경로
import codecs

def save_metrics(df, cfg, model_name, output_dir):
    # 데이터 전처리
    print("\nData preprocessing stats before:")
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # 감정 라벨 소문자로 변환
    df['emotion'] = df['emotion'].str.lower()
    df[f'predicted_emotion_{model_name}'] = df[f'predicted_emotion_{model_name}'].str.lower()
    
    # 결측치와 중복 제거
    df = df.dropna(subset=['emotion', f'predicted_emotion_{model_name}'])
    df = df.drop_duplicates()
    
    print("\nData preprocessing stats after:")
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # NaN 값을 default_emotion으로 채우기
    default_emotion = cfg.data.default_emotion
    
    # config에서 정의된 라벨 목록 사용
    labels = cfg.data.datasets[cfg.data.name].labels
    
    # 매핑 딕셔너리 생성
    mapping_dict = {
        'unknown': default_emotion,
        'indifference': default_emotion,
        'confusion': default_emotion,
    }
    
    # 실제 라벨과 예측 라벨 추출하고 문자열로 변환
    y_true = df['emotion'].fillna(default_emotion).astype(str)
    y_pred = df[f'predicted_emotion_{model_name}'].fillna(default_emotion).astype(str)
    
    # 예측 라벨을 config의 라벨로 매핑
    y_pred = y_pred.map(lambda x: mapping_dict.get(x, x) if x not in labels else x)
    
    print(f"\nLabels information:")
    print(f"Config labels: {labels}")
    print(f"Unique values in true labels: {y_true.unique()}")
    print(f"Unique values in predicted labels after mapping: {y_pred.unique()}")
    
    # 분류 보고서 생성 (zero_division=0 추가)
    report_dict = classification_report(y_true, y_pred, 
                                      labels=labels,
                                      output_dict=True, 
                                      zero_division=0)
    report_str = classification_report(y_true, y_pred, 
                                     labels=labels,
                                     zero_division=0)
    
    # 분류 보고서 텍스트 파일로 저장
    with open(output_dir / f'classification_report_{model_name}.txt', "w") as file:
        file.write(report_str)
    
    # Classification Report를 DataFrame으로 변환하여 CSV로 저장
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / f'classification_report_{model_name}.csv')
    
    # 혼동 행렬 계산 (명시적으로 라벨 지정)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    # 결과 저장을 위한 경로 설정
    
    metrics_path = output_dir / f'metrics_{model_name}.json'
    
    # 메트릭 결과 저장
    metrics_result = {
        "overall": {
            "accuracy": report_dict.get('accuracy', 0.0),
            "macro_avg": report_dict.get('macro avg', {}),
            "weighted_avg": report_dict.get('weighted avg', {})
        },
        "per_class": {
            label: report_dict[label] for label in report_dict.keys()
            if label not in ['accuracy', 'macro avg', 'weighted avg']
        }
    }
    
    # JSON 파일로 저장
    with open(metrics_path, 'w') as f:
        json.dump(metrics_result, f, indent=2)
        
    # 두 개의 subplot으로 혼동 행렬 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # 원본 혼동 행렬
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax1)
    ax1.set_title(f'Confusion Matrix - {model_name}')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 정규화된 혼동 행렬
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax2)
    ax2.set_title(f'Normalized Confusion Matrix - {model_name}')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 두 subplot 모두에 대해 라벨 회전 및 정렬 조정
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # 그래프가 잘리지 않도록 레이아웃 조정
    plt.tight_layout()
    
    # 더 높은 해상도로 저장
    plt.savefig(output_dir / f'confusion_matrix_{model_name}.png', 
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    return metrics_result, cm

def map_unknown_emotion(emotion: str, labels: list, mapping_dict: dict = None, cfg: dict = None) -> str:
    """알 수 없는 감정을 매핑"""
    if mapping_dict is None:
        mapping_dict = {
            'happy': 'joy',
            'happiness': 'joy',
            'anxious': 'fear',
            'anxiety': 'fear',
            'worried': 'fear',
            'mad': 'anger',
            'frustrated': 'anger',
            'depressed': 'sadness',
            'disappointed': 'sadness',
            'disgusted': 'disgust',
            'ashamed': 'shame',
            'embarrassed': 'shame',
            'regret': 'guilt',
            'remorse': 'guilt',
            'unknown': 'sadness'
        }
    
    mapped_emotion = mapping_dict.get(emotion.lower(), cfg.data.default_emotion)
    print(f"Mapping emotion: {emotion} -> {mapped_emotion}")  # 매핑 로그 추가
    return mapped_emotion

def retry_emotion_prediction(text: str, labels: list, client, model: str, temperature: float, seed: int) -> dict:
    """감정 재예측 시도"""
    # 더 명확한 프롬프트와 제약조건 추가
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": f"Strictly classify the emotion as one of these ONLY: {', '.join(labels)}",
            "parameters": {
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": labels,  # 가능한 값을 명시적으로 제한
                        "description": f"Must be one of: {', '.join(labels)}"
                    },
                    "confidence_score": {
                        "type": "number",
                        "description": "confidence score of the emotion: 0.0-1.0"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "explain why the emotion is decided"
                    },
                },
                "required": ["emotion", "confidence_score", "explanation"]
            }
        }
    }]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You MUST classify the emotion as one of these ONLY: {', '.join(labels)}. Do not use any other emotion words."},
                {"role": "user", "content": text}
            ],
            tools=tools,
            seed=seed,
            temperature=temperature,
            timeout=10
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            return json.loads(tool_call.function.arguments)
    except Exception as e:
        print(f"Error in retry prediction: {e}")
    
    return {"emotion": "sadness", "confidence_score": 0.5, "explanation": "Failed to get valid prediction"}

def load_data(cfg):
    dataset_cfg = cfg.data.datasets[cfg.data.name]
    labels = dataset_cfg.labels
    df = pd.read_csv(
        dataset_cfg.path,
        sep=dataset_cfg.separator,
        on_bad_lines='skip',
        encoding='utf-8',
        engine='python',
        quoting=3,
        dtype=str
    )

    required_columns = dataset_cfg.required_columns
    
    if not all(col in df.columns for col in required_columns):
        df.columns = [col.split(dataset_cfg.separator)[0] for col in df.columns]
    
    df = df[required_columns]
    df.rename(columns={'SIT': 'text', 'EMOT': 'emotion'}, inplace=True)
    df['emotion'] = df['emotion'].map(lambda x: labels[int(x)-1] if x.isdigit() else 'undefined')
    
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def preprocess_text(text):
        if not isinstance(text, str):
            return text
        text = ' '.join(text.split())
        text = text.lower()
        return text
    
    print("\nData preprocessing stats before:")
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # 결측치 데이터 저장
    missing_data = df[df.isna().any(axis=1)].copy()
    if not missing_data.empty:
        missing_data['removal_reason'] = 'missing_value'
        missing_data['status'] = 'removed'
        missing_data.to_csv(Path(cfg.general.output_path) / f'missing_data.csv', index=True)
        print(f"\nSaved {len(missing_data)} rows with missing values")
    
    # 결측치 제거
    df_no_missing = df.dropna()
    
    # 텍스트 전처리 적용
    df_no_missing['text'] = df_no_missing['text'].apply(preprocess_text)
    
    # exclude_phrases 처리
    excluded_rows = pd.DataFrame()
    if cfg.data.exclude_phrases:
        exclude_phrases = list(cfg.data.exclude_phrases)
        original_len = len(df_no_missing)
        
        # 제외될 행들을 미리 저장
        for phrase in exclude_phrases:
            # 텍스트 길이가 exclude_length 미만인 경우에만 필터링
            matched_rows = df_no_missing[
                (df_no_missing['text'].str.contains(phrase, case=False, na=False)) & 
                (df_no_missing['text'].str.len() < cfg.data.exclude_length)
            ].copy()
            
            if not matched_rows.empty:
                matched_rows['removal_reason'] = f'contains_phrase: {phrase}'
                matched_rows['status'] = 'removed'
                excluded_rows = pd.concat([excluded_rows, matched_rows])
        
        # 제외 구문 포함된 행 제거 (길이 조건 포함)
        for phrase in exclude_phrases:
            df_no_missing = df_no_missing[
                ~((df_no_missing['text'].str.contains(phrase, case=False, na=False)) & 
                  (df_no_missing['text'].str.len() < cfg.data.exclude_length))
            ]
        
        filtered_len = len(df_no_missing)
        print(f"\nRemoved {original_len - filtered_len} rows containing excluded phrases (length < {cfg.data.exclude_length})")
        
        # 제외된 행 저장
        if not excluded_rows.empty:
            excluded_rows.to_csv(Path(cfg.general.output_path) / f'excluded_phrases_data.csv', index=True)
            print(f"Saved {len(excluded_rows)} rows containing excluded phrases")
    
    # 중복 데이터 처리
    duplicates_all = df_no_missing[df_no_missing.duplicated(subset=['text'], keep=False)].copy()
    if not duplicates_all.empty:
        # 중복 데이터에 그룹 ID 추가
        duplicates_all['duplicate_group'] = duplicates_all.groupby('text').ngroup()
        
        # 각 그룹의 첫 번째 항목은 유지되고 나머지는 제거됨을 표시
        duplicates_all['status'] = 'removed'
        duplicates_all.loc[~duplicates_all.duplicated(subset=['text'], keep='first'), 'status'] = 'kept'
        
        # duplicate_group으로 정렬하고, 같은 그룹 내에서는 status로 정렬 (kept가 먼저 오도록)
        duplicates_all = duplicates_all.sort_values(['duplicate_group', 'status'], 
                                                  ascending=[True, True])
        
        # 중복 데이터 저장 (유지된 것과 제거된 것 모두)
        duplicates_all.to_csv(Path(cfg.general.output_path) / f'duplicate_data.csv', index=True)
        print(f"\nSaved {len(duplicates_all)} rows of duplicate data (including kept and removed)")
        print(f"- Kept: {len(duplicates_all[duplicates_all['status'] == 'kept'])} rows")
        print(f"- Removed: {len(duplicates_all[duplicates_all['status'] == 'removed'])} rows")
    
    # 중복 제거 (첫 번째 항목 유지)
    df_final = df_no_missing.drop_duplicates(subset=['text'], keep='first')
    
    # 감정 클래스별 통계 계산
    emotion_stats = {}
    for emotion in labels:
        initial_count = len(df[df['emotion'] == emotion])
        excluded_count = len(excluded_rows[excluded_rows['emotion'] == emotion])
        duplicates_removed = len(duplicates_all[(duplicates_all['emotion'] == emotion) & 
                                              (duplicates_all['status'] == 'removed')]) if not duplicates_all.empty else 0
        final_count = len(df_final[df_final['emotion'] == emotion])
        
        emotion_stats[emotion] = {
            'initial_count': initial_count,
            'excluded_count': excluded_count,
            'duplicates_removed': duplicates_removed,
            'final_count': final_count,
            'total_removed': excluded_count + duplicates_removed,
            'removal_percentage': ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
        }
    
    # 제거된 데이터 통계 저장
    removal_stats = {
        'total_initial_samples': len(df),
        'total_final_samples': len(df_final),
        'removed_missing_values': len(missing_data),
        'removed_excluded_phrases': len(excluded_rows),
        'duplicate_stats': {
            'total_duplicates': len(duplicates_all) if not duplicates_all.empty else 0,
            'kept_duplicates': len(duplicates_all[duplicates_all['status'] == 'kept']) if not duplicates_all.empty else 0,
            'removed_duplicates': len(duplicates_all[duplicates_all['status'] == 'removed']) if not duplicates_all.empty else 0
        },
        'excluded_phrases_list': list(cfg.data.exclude_phrases) if cfg.data.exclude_phrases else [],
        'emotion_class_stats': emotion_stats
    }
    
    with open(Path(cfg.general.output_path) / f'removal_stats.json', 'w', encoding='utf-8') as f:
        json.dump(removal_stats, f, indent=2, ensure_ascii=False)
    
    print("\nData preprocessing stats after:")
    print(f"Total samples: {len(df_final)}")
    print(f"Total removed samples:")
    print(f"- Missing values: {len(missing_data)}")
    print(f"- Excluded phrases: {len(excluded_rows)}")
    print(f"- Duplicates: {len(duplicates_all[duplicates_all['status'] == 'removed']) if not duplicates_all.empty else 0}")
    print("\nEmotion class statistics:")
    for emotion, stats in emotion_stats.items():
        print(f"\n{emotion}:")
        print(f"- Initial count: {stats['initial_count']}")
        print(f"- Excluded: {stats['excluded_count']}")
        print(f"- Duplicates removed: {stats['duplicates_removed']}")
        print(f"- Final count: {stats['final_count']}")
        print(f"- Removal percentage: {stats['removal_percentage']:.2f}%")
    
    return df_final, labels

def clean_json_response(response_text: str) -> str:
    """JSON 응답 정리"""
    try:
        print("\nRaw response:", repr(response_text))  # 실제 응답 확인
        
        # 1. 코드 블록 제거
        if "```" in response_text:
            pattern = r"```(?:json)?(.*?)```"
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                response_text = matches[0]
                print("After code block removal:", repr(response_text))
        
        # 2. JSON 블록 찾기
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            raise ValueError("No valid JSON found in response")
            
        json_text = response_text[json_start:json_end]
        print("Extracted JSON:", repr(json_text))
        
        # 3. 모든 공백 문자 정리 (더 강력한 방식)
        # 먼저 모든 공백 문자를 일반 공백으로 변환
        json_text = re.sub(r'[\n\r\t\s]+', ' ', json_text)
        print("After whitespace cleanup:", repr(json_text))
        
        # 4. JSON 파싱 및 재직렬화
        parsed = json.loads(json_text)
        final_json = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
        print("Final JSON:", repr(final_json))
        
        return final_json
        
    except Exception as e:
        print(f"Error in clean_json_response: {e}")
        print(f"Original response: {response_text}")
        raise

def get_model_response(text: str, labels: list, client, model: str, temperature: float, cfg, llm=None, logger=None, rag=None, tools=None) -> dict:
    """모델 타입에 따라 적절한 응답 방식 선택"""
    try:
        # 먼저 template 정의
        if cfg.model.use_template:
            model_key = model.lower()
            template = cfg.prompt.get(model_key)
            if not template:
                if logger:
                    logger.info(f"No specific template found for model {model_key}, using default template")
                template = cfg.prompt.emotion_fewshot
        else:
            template = cfg.prompt.emotion

        # RAG 또는 일반 프롬프트 생성
        if cfg.model.use_rag and rag:
            similar_examples = rag.get_similar_examples(text, k=cfg.rag.k_examples)
            prompt_template = str(rag.get_rag_prompt(text, similar_examples))
            if logger:
                safe_log(logger, 'debug', f"RAG prompt: {prompt_template}")
        else:
            prompt_template = f"""{template}

Text: {text}

IMPORTANT: Return ONLY a single-line JSON object without any formatting."""

        if logger:
            safe_log(logger, 'debug', f"Raw prompt: {prompt_template}")

        try:
            if cfg.model.type == "ollama":
                response = llm.invoke(prompt_template)
            elif cfg.model.type == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{template}\n\nText: {text}"
                        }
                    ],
                    temperature=temperature
                )
                response = response.content[0].text
            else:
                # OpenAI, Upstage 등은 chat completion 사용
                messages = [
                    {"role": "system", "content": template},
                    {"role": "user", "content": text}
                ]
                
                if cfg.model.function_calling and cfg.model.type != "ollama":
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        tools=tools,
                        seed=int(cfg.general.seed)
                    )
                    # Function calling 응답 처리
                    if response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]
                        result = json.loads(tool_call.function.arguments)
                        if logger:
                            logger.debug(f"Function call result: {result}")
                        return result
                else:
                    # 일반 응답
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
                response = response.choices[0].message.content

            if logger:
                safe_log(logger, 'debug', f"Raw response: {response}")

            # JSON 파싱
            result = clean_json_response(response)
            if isinstance(result, str):
                result = json.loads(result)

            if logger:
                safe_log(logger, 'info', f"Processed response: {result}")
            return result

        except Exception as e:
            if logger:
                safe_log(logger, 'error', f"Error generating response: {e}")
            raise

    except Exception as e:
        if logger:
            safe_log(logger, 'error', f"Error in get_model_response: {e}")
        raise

def get_prompt(cfg, text: str, labels: list, model_name: str) -> str:
    """프롬프트 템플릿 선택 및 생성"""
    if cfg.model.use_template:
        # 모델 이름에 따라 프롬프트 선택
        if 'llama' in model_name.lower():
            template = cfg.prompt.llama
        elif 'qwen' in model_name.lower():
            template = cfg.prompt.qwen
        else:
            # 기본 프롬프트
            template = cfg.prompt.target_prompt
        return template
    else:
        return f"""Analyze the emotional content of the following text and classify it as one of: {', '.join(labels)}

Text: {text}

Provide your analysis in a structured format."""

def save_prompt_response(text: str, prompt, response: dict, output_dir: Path, index: int):
    """프롬프트와 응답을 로그로 저장"""
    # output_dir 내부에 logs 폴더 생성
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 프롬프트 객체를 문자열로 변환
    if hasattr(prompt, 'partial_variables'):
        # RAG 프롬프트의 경우
        prompt_str = prompt.partial_variables.get('context', str(prompt))
    else:
        # 일반 프롬프트의 경우
        prompt_str = str(prompt)
    
    log_entry = {
        "index": index,
        "input_text": text,
        "prompt": prompt_str,
        "response": response
    }
    
    log_file = log_dir / "prompt_response_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def format_prompt_for_display(prompt, text):
    """프롬프트를 보기 좋게 포맷팅"""
    output = "\n" + "="*50 + "\n"
    output += "📝 PROMPT DETAILS\n" + "="*50 + "\n\n"
    
    # 입력 텍스트
    output += "📌 Input Text:\n"
    output += f"{text}\n\n"
    
    # RAG 컨텍스트가 있는 경우
    if hasattr(prompt, 'partial_variables') and 'context' in prompt.partial_variables:
        context = prompt.partial_variables['context']
        
        # 유사 예제 추출
        if "Similar examples for reference:" in context:
            output += "🔍 Similar Examples:\n"
            examples = context.split("Similar examples for reference:\n")[1].split("\nRemember")[0]
            output += f"{examples}\n"
        
        # 지침 추출
        if "Remember to:" in context:
            output += "📋 Guidelines:\n"
            guidelines = context.split("Remember to:")[1].split("Format your response")[0]
            output += f"{guidelines}\n"
    
    output += "-"*50 + "\n"
    return output

class SafeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if sys.platform == 'win32':
            if stream is None:
                self.stream = codecs.getwriter('utf-8')(sys.stdout.buffer)
            elif hasattr(stream, 'buffer'):
                self.stream = codecs.getwriter('utf-8')(stream.buffer)

    def emit(self, record):
        try:
            msg = self.format(record)
            try:
                if sys.platform == 'win32' and hasattr(self.stream, 'buffer'):
                    self.stream.buffer.write(msg.encode('utf-8'))
                    self.stream.buffer.write(self.terminator.encode('utf-8'))
                else:
                    self.stream.write(msg)
                    self.stream.write(self.terminator)
                self.flush()
            except UnicodeEncodeError:
                try:
                    safe_msg = msg.encode('utf-8', errors='replace').decode('utf-8')
                    if sys.platform == 'win32' and hasattr(self.stream, 'buffer'):
                        self.stream.buffer.write(safe_msg.encode('utf-8'))
                        self.stream.buffer.write(self.terminator.encode('utf-8'))
                    else:
                        self.stream.write(safe_msg)
                        self.stream.write(self.terminator)
                    self.flush()
                except:
                    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                    sys.stderr.write(safe_msg + self.terminator)
                    self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(cfg, output_dir: Path):
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = output_dir / cfg.general.logging.log_path
    log_dir.mkdir(parents=True, exist_ok=True)

    # 로그 파일 경로
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 로거 설정
    logger = logging.getLogger('emotion_analysis')
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거
    logger.handlers.clear()

    # Windows 환경에서 UTF-8 출력을 위한 설정
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        # 파일 핸들러 설정
        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)

        # 콘솔 핸들러 설정
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # INFO 레벨로 제한
        
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    except Exception as e:
        print(f"Error setting up logging: {e}")
        # 기본 콘솔 로거만 설정
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def safe_log(logger, level: str, message: str):
    """안전한 로깅 함수"""
    try:
        if not isinstance(message, str):
            message = str(message)
        
        # ASCII로 변환 (한글과 특수문자 제거)
        message = message.encode('ascii', errors='ignore').decode('ascii')
            
        # 메시지 길이 제한
        if len(message) > 1000:
            message = message[:997] + "..."
        
        getattr(logger, level)(message)
            
    except Exception as e:
        print(f"Logging error: {e}")

def get_output_dir_name(model_name: str, cfg) -> str:
    """설정에 따른 출력 폴더명 생성"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [timestamp, model_name]
    
    # 각 설정이 True일 때만 이름에 추가
    if cfg.model.use_template:
        name_parts.append("template")
    if cfg.model.use_rag:
        name_parts.append("rag")
    if cfg.model.function_calling:
        name_parts.append("function_calling")
    
    return "_".join(name_parts)

def initialize_models(cfg):
    """LLM과 RAG 모델 초기화"""
    llm = None
    rag = None
    
    if cfg.model.type == "ollama":
        llm = OllamaLLM(
            model=cfg.model.name,
            temperature=cfg.model.temperature
        )
    elif cfg.model.type == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        llm = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif cfg.model.type == "upstage":
        base_url = "https://api.upstage.ai/v1/solar"
        api_key = os.getenv("UPSTAGE_API_KEY")
        llm = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif cfg.model.type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        llm = Anthropic(api_key=api_key)
    
    if cfg.model.use_rag:
        rag = EmotionRAG(cfg)
    
    return llm, rag

def sanitize_model_name(model_name: str) -> str:
    """모델 이름을 파일명으로 사용 가능하도록 정리"""
    return re.sub(r'[^\w\-\.]', '_', model_name.lower())

@hydra.main(version_base="1.2", config_path='config', config_name='llm')
def main(cfg):
    # 모델 초기화를 메인 로직 시작 전에 수행
    llm, rag = initialize_models(cfg)
    
    # 모든 config 값들을 Python 기본 타입으로 변환
    model_type = str(cfg.model.type)
    model = str(cfg.model.name)
    temperature = float(cfg.model.temperature)
    
    # 안전한 모델명 생성
    model_name = sanitize_model_name(model)
    print(f"Using sanitized model name: {model_name}")
    
    # 출력 디렉토리 설정
    output_dir_name = get_output_dir_name(model_name, cfg)
    output_dir = Path('outputs') / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # cfg의 output_path 업데이트
    cfg.general.output_path = str(output_dir)
    
    
    # 데이터 로드 및 라벨 준비
    df_isear, labels = load_data(cfg)
    labels = list(map(str, labels))
    
    # n_samples 설정
    n_samples = len(df_isear) if cfg.data.n_samples == -1 else min(cfg.data.n_samples, len(df_isear))
    print(f"Processing {n_samples} samples out of {len(df_isear)} total samples")
    
    # 처음 n_samples 개만 선택
    df_isear = df_isear.head(n_samples)
    
    # 출력 파일 경로 설정 (디렉토리는 이미 생성됨)
    output_path = Path(cfg.general.output_path) / f'dataset-{cfg.data.name}_model-{model_name}.csv'

    # 클라이언트 초기화
    if model_type == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key  
        )
    elif model_type == "upstage":
        base_url = "https://api.upstage.ai/v1/solar"
        api_key = os.getenv("UPSTAGE_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key  
        )
    elif model_type == "ollama":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
        client = OpenAI(
            base_url=base_url,
            api_key=api_key  
        )
    elif model_type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        client = Anthropic(api_key=api_key)
        print("Anthropic Claude model initialized")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Function calling 도구 정의
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Predict the emotion of the text",
                "parameters": {
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "enum": list(labels),  # 명시적으로 list로 변환
                            "description": f"Classify the emotion of the text as one of given labels: {', '.join(labels)}"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "confidence score of the emotion"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "explain why the emotion is decided"
                        },
                    },
                    "required": ["emotion", "confidence_score", "explanation"],
                    "type": "object"
                }
            }
        }
    ]

    # 잘못된 감정 예측 통계 초기화
    invalid_predictions = {
        "total": 0,
        "invalid_samples": [],
        "by_true_label": {str(label): 0 for label in labels}
    }

    # RAG 초기화 (use_rag가 true일 때)
    rag = None
    if cfg.model.use_rag:
        rag = EmotionRAG(cfg)
        rag.create_index(df_isear)

    # 로깅 설정
    logger = setup_logging(cfg, output_dir)

    for index, row in tqdm(df_isear.iterrows(), total=len(df_isear)):
        try:
            if rag:
                rag.exclude_index(index)  # 현재 인덱스 제외
                
            result = get_model_response(
                str(row.text),
                labels,
                client,
                model,
                temperature,
                cfg,
                llm=llm,
                logger=logger,
                rag=rag,
                tools=tools  # tools 전달
            )
            
            # 예측된 감정이 유효한지 먼저 확인
            original_emotion = result["emotion"].lower()
            
            # 잘못된 예측 확인 및 기록
            if original_emotion not in [l.lower() for l in labels]:
                print(f"\nInvalid prediction found at index {index}")
                print(f"Text: {row.text[:100]}...")
                print(f"True emotion: {row.emotion}")
                print(f"Predicted: {original_emotion}")
                
                # 잘못된 예측 통계 업데이트
                invalid_predictions["total"] += 1
                invalid_predictions["by_true_label"][str(row.emotion)] += 1
                
                # 잘못된 예측 샘플 저장 (모든 필요한 필드 포함)
                invalid_sample = {
                    "index": int(index),
                    "text": str(row.text),
                    "true_emotion": str(row.emotion),
                    "predicted_emotion": str(original_emotion),
                    "confidence": float(result.get("confidence_score", 0.0)),
                    "explanation": str(result.get("explanation", "No explanation"))
                }
                invalid_predictions["invalid_samples"].append(invalid_sample)
                print(f"Total invalid samples so far: {len(invalid_predictions['invalid_samples'])}")
                
                # 재시도 또는 매핑 로직
                if getattr(cfg.general, 'retry_invalid_predictions', False):
                    print("Attempting retry...")  # 재시도 시작
                    try:
                        retry_result = retry_emotion_prediction(
                            str(row.text), labels, client, model, temperature, int(cfg.general.seed)
                        )
                        print(f"Retry result: {retry_result}")  # 재시도 결과 출력
                        result["emotion"] = retry_result.get("emotion", "unknown")
                        result["confidence_score"] = float(retry_result.get("confidence_score", 0.0))
                        result["explanation"] = str(retry_result.get("explanation", "No explanation from retry"))
                        
                        # 여전히 잘못된 경우 매핑
                        if result["emotion"].lower() not in [l.lower() for l in labels]:
                            print(f"Retry emotion still invalid: {result['emotion']}, applying mapping")
                            result["emotion"] = map_unknown_emotion(result["emotion"], labels, cfg=cfg)
                    except Exception as e:
                        print(f"Error in retry for row {index}: {e}")
                        result["emotion"] = map_unknown_emotion(result["emotion"], labels, cfg=cfg)
                else:
                    print("Applying direct mapping")
                    result["emotion"] = map_unknown_emotion(original_emotion, labels, cfg=cfg)

            # 결과 저장
            df_isear.at[index, f'predicted_emotion_{model_name}'] = result["emotion"]
            df_isear.at[index, f'confidence_score_{model_name}'] = result["confidence_score"]
            df_isear.at[index, f'explanation_{model_name}'] = result["explanation"]
            
            # 예측 결과를 로그 파일에도 저장
            result_log = f'''
=== Prediction Result (Index: {index}) ===
Text: {row.text}
Ground Truth: {row.emotion}
Predicted Emotion: {result['emotion']}
Confidence Score: {result['confidence_score']}
Explanation: {result['explanation']}
{'='*50}
'''
            logger.info(result_log)  # 로그 파일에 저장
            
            # 콘솔 출력용 (기존 print문 대체)
            print(result_log)

            if index % 500 == 0:
                df_isear.to_csv(output_path, index=False)
            
            # 로깅
            logger.debug(f"Raw response: {result}")
            logger.info(f"Processed response: {result}")

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            # 오류 발생 시 기본값 저장
            df_isear.at[index, f'predicted_emotion_{model_name}'] = "unknown"
            df_isear.at[index, f'confidence_score_{model_name}'] = 0.0
            df_isear.at[index, f'explanation_{model_name}'] = f"Error: {str(e)}"
            continue

    # 최종 결과 저장
    df_isear.to_csv(output_path, index=False)
    
    # 잘못된 예측 통계 저장
    invalid_stats = {
        "total_samples": n_samples,
        "invalid_count": invalid_predictions["total"],
        "invalid_percentage": (invalid_predictions["total"] / n_samples) * 100,
        "by_true_label": invalid_predictions["by_true_label"],
        "invalid_samples": invalid_predictions["invalid_samples"]
    }
    
    print(f"\nInvalid prediction statistics:")
    print(f"Total invalid: {invalid_stats['invalid_count']}")
    print(f"Invalid percentage: {invalid_stats['invalid_percentage']:.2f}%")
    print(f"Number of invalid samples collected: {len(invalid_stats['invalid_samples'])}")
    
    # 통계 저장
    save_prediction_stats(df_isear, invalid_stats, output_dir, model_name, labels)
    
    # 평가 메트릭 계산 및 저장
    report, cm = save_metrics(df_isear, cfg, model_name, output_dir)
    
    # 주요 메트릭 출력
    print("\nClassification Report Summary:")
    print(f"Accuracy: {report['overall']['accuracy']:.4f}")
    print("\nPer-class metrics:")
    for label, metrics in report['per_class'].items():
        print(f"{label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")

def save_prediction_stats(df, invalid_stats, output_dir, model_name, labels):
    """예측 통계 저장"""
    try:
        # 통계 파일 저장
        stats_file = output_dir / f'prediction_stats_{model_name}.json'
        with open(stats_file, 'w', encoding='ascii') as f:
            json.dump(invalid_stats, f, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"Error saving prediction stats: {e}")

def metric():
    cfg = OmegaConf.load("config/llm.yaml")
    
    # 경로를 Path 객체로 변환하여 올바른 경로 구성
    base_path = Path("D:/dev/isp_rag_cluster/outputs")
    
    # 실제 존재하는 디렉토리 확인
    output_folders = [f for f in base_path.iterdir() if f.is_dir()]
    if not output_folders:
        raise FileNotFoundError("No output folders found")
    
    # 가장 최근 폴더 선택 (타임스탬프 기준)
    latest_folder = max(output_folders, key=lambda x: x.stat().st_mtime)
    latest_folder = Path("outputs/20250228_122946_llama3.2/dataset-isear_model-llama3.2") #Path("outputs/20250228_000911/dataset-isear_model-gpt-3.5-turbo")
    # 파일 경로 구성
    model_name = "llama3.2"  # 또는 다른 모델명
    model_name = latest_folder.name.split("_model-")[1]
    file_name = f"dataset-isear_model-{model_name}.csv"
    latest_folder = latest_folder.parent
    file_path = latest_folder / file_name
    
    print(f"Looking for file at: {file_path}")  # 디버깅용
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # 파일 읽기
    df_isear = pd.read_csv(file_path)
    
    # 메트릭 저장
    save_metrics(df_isear, cfg, model_name, latest_folder)

if __name__ == "__main__":
    main()
    #metric()
    
