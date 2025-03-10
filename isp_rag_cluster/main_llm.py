from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import json
import re
from ollama import chat, ChatResponse
from anthropic import Anthropic  # Import Anthropic package
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
from langchain_community.embeddings import HuggingFaceEmbeddings  # New import path
import codecs

# Add global flag for prompt logging
_PROMPT_LOGGED = False
_INPUT_LOGGED = False  # 새로 추가

def save_metrics(df, cfg, model_name, output_dir):
    """Save prediction statistics and metrics"""
    print("\nData preprocessing statistics (before):")
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Convert emotion labels to lowercase
    df['emotion'] = df['emotion'].str.lower()
    df[f'predicted_emotion_{model_name}'] = df[f'predicted_emotion_{model_name}'].str.lower()
    
    # Remove missing values and duplicates
    df = df.dropna(subset=['emotion', f'predicted_emotion_{model_name}'])
    df = df.drop_duplicates()
    
    # Use label list defined in config (convert ListConfig to regular list)
    labels = list(cfg.data.datasets[cfg.data.name].labels)
    default_emotion = str(cfg.data.default_emotion)
    
    # Get unique predicted labels
    predicted_labels = df[f'predicted_emotion_{model_name}'].unique()
    
    # Find invalid predictions (not in defined labels)
    invalid_predictions = df[~df[f'predicted_emotion_{model_name}'].isin(labels)]
    
    # Create invalid predictions statistics
    invalid_stats = {
        "total_samples": len(df),
        "invalid_count": len(invalid_predictions),
        "invalid_percentage": (len(invalid_predictions) / len(df)) * 100 if len(df) > 0 else 0,
        "invalid_labels": [label for label in predicted_labels if label not in labels],
        "samples": []
    }
    
    # Add detailed information for each invalid prediction
    for _, row in invalid_predictions.iterrows():
        invalid_stats["samples"].append({
            "text": row['text'],
            "true_emotion": row['emotion'],
            "predicted_emotion": row[f'predicted_emotion_{model_name}'],
            "confidence_score": row.get(f'confidence_score_{model_name}', None),
            "explanation": row.get(f'explanation_{model_name}', None)
        })
    
    # Save invalid predictions statistics
    invalid_file = output_dir / f'invalid_predictions_{model_name}.json'
    with open(invalid_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_stats, f, indent=2, ensure_ascii=False)
    
    # Select only valid samples (predictions in defined labels)
    valid_samples = df[df[f'predicted_emotion_{model_name}'].isin(labels)]
    
    print("\nPrediction validation statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Valid predictions: {len(valid_samples)}")
    print(f"Invalid predictions: {len(invalid_predictions)}")
    print(f"Invalid labels found: {invalid_stats['invalid_labels']}")
    
    # Extract actual and predicted labels for valid samples only
    y_true = valid_samples['emotion']
    y_pred = valid_samples[f'predicted_emotion_{model_name}']
    
    print(f"\nLabel information:")
    print(f"Defined labels: {labels}")
    print(f"Actual label unique values: {y_true.unique()}")
    print(f"Predicted label unique values (valid only): {y_pred.unique()}")
    
    # Generate classification report for valid samples
    report_dict = classification_report(y_true, y_pred, 
                                      labels=labels,
                                      output_dict=True, 
                                      zero_division=0)
    report_str = classification_report(y_true, y_pred, 
                                     labels=labels,
                                     zero_division=0)
    
    # Save classification report as text file
    with open(output_dir / f'classification_report_{model_name}.txt', "w", encoding='utf-8') as file:
        file.write(f"Total samples: {len(df)}\n")
        file.write(f"Valid samples: {len(valid_samples)}\n")
        file.write(f"Invalid predictions: {len(invalid_predictions)}\n")
        file.write(f"Invalid labels found: {', '.join(invalid_stats['invalid_labels'])}\n\n")
        file.write("=== Classification Report (Valid Samples Only) ===\n")
        file.write(report_str)
    
    # Save classification report as CSV with transpose
    report_df = pd.DataFrame(report_dict).round(4)
    report_df = report_df.drop(['accuracy'], errors='ignore')  # accuracy 행이 있다면 제거
    report_df = report_df.transpose()  # 행과 열을 전치
    report_df.index.name = 'class'  # 인덱스 이름을 'metric'으로 변경
    report_df.to_csv(output_dir / f'classification_report_{model_name}.csv')
    
    # Save metrics result
    metrics_result = {
        "overall": {
            "accuracy": float(report_dict['accuracy']),
            "macro_avg": report_dict.get('macro avg', {}),
            "weighted_avg": report_dict.get('weighted avg', {})
        },
        "per_class": {
            label: report_dict[label] for label in report_dict.keys()
            if label not in ['accuracy', 'macro avg', 'weighted avg']
        },
        "invalid_predictions_info": {
            "total": len(invalid_predictions),
            "percentage": (len(invalid_predictions) / len(df)) * 100 if len(df) > 0 else 0,
            "invalid_labels": invalid_stats['invalid_labels']
        }
    }
    
    # Calculate confusion matrix for valid samples only
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    # Plot confusion matrix in two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Original confusion matrix
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax1)
    ax1.set_title(f'Confusion Matrix - {model_name}\n(Valid samples only: {len(valid_samples)})')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax2)
    ax2.set_title(f'Normalized Confusion Matrix - {model_name}\n(Valid samples only: {len(valid_samples)})')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{model_name}.png', 
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    return metrics_result, cm


def retry_emotion_prediction(text: str, labels: list, client, model: str, temperature: float, seed: int, cfg: dict) -> dict:
    """Retry emotion prediction attempt"""
    # 특정 모델들은 function calling을 지원하지 않음

    
    use_function_call = any(model_name.lower() in model.lower() for model_name in cfg.prompt.function_call_models)
    
    try:
        if use_function_call and cfg.model.function_calling:
            tools = [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": f"Strictly classify the emotion as one of these ONLY: {', '.join(labels)}",
                    "parameters": {
                        "properties": {
                            "emotion": {
                                "type": "string",
                                "enum": labels,
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
        else:
            # Function calling을 지원하지 않는 모델을 위한 일반 응답
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You MUST classify the emotion as one of these ONLY: {', '.join(labels)}. Do not use any other emotion words."},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
                timeout=10
            )
            return clean_json_response(response.choices[0].message.content)
            
    except Exception as e:
        print(f"Error in retry prediction: {e}")
    
    return {"emotion": "unknown", "confidence_score": 0.0, "explanation": "Failed to get valid prediction"}

def load_data(cfg, logger=None):
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
    
    def preprocess_text(text):
        if not isinstance(text, str):
            return text
        text = ' '.join(text.split())
        text = text.lower()
        return text
    
    # Log message construction
    log_messages = []
    log_messages.append("\nData preprocessing stats before:")
    log_messages.append(f"Total samples: {len(df)}")
    log_messages.append(f"Missing values:\n{df.isna().sum()}")
    log_messages.append(f"Duplicates: {df.duplicated().sum()}")
    
    # Create subfolder for saving data-related files
    data_dir = Path(cfg.general.output_path) / "data_stats"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save missing data
    missing_data = df[df.isna().any(axis=1)].copy()
    if not missing_data.empty:
        missing_data['removal_reason'] = 'missing_value'
        missing_data['status'] = 'removed'
        missing_data.to_csv(data_dir / 'missing_data.csv', index=True)
        log_messages.append(f"\nSaved {len(missing_data)} rows with missing values")
    
    # Remove missing values
    df_no_missing = df.dropna()
    
    # Apply text preprocessing
    df_no_missing['text'] = df_no_missing['text'].apply(preprocess_text)
    
    # exclude_phrases processing
    excluded_rows = pd.DataFrame()
    if cfg.data.exclude_phrases:
        exclude_phrases = list(cfg.data.exclude_phrases)
        original_len = len(df_no_missing)
        
        # Save rows to exclude in advance
        for phrase in exclude_phrases:
            matched_rows = df_no_missing[
                (df_no_missing['text'].str.contains(phrase, case=False, na=False)) & 
                (df_no_missing['text'].str.len() < cfg.data.exclude_length)
            ].copy()
            
            if not matched_rows.empty:
                matched_rows['removal_reason'] = f'contains_phrase: {phrase}'
                matched_rows['status'] = 'removed'
                excluded_rows = pd.concat([excluded_rows, matched_rows])
        
        # Remove rows containing excluded phrases (including length condition)
        for phrase in exclude_phrases:
            df_no_missing = df_no_missing[
                ~((df_no_missing['text'].str.contains(phrase, case=False, na=False)) & 
                  (df_no_missing['text'].str.len() < cfg.data.exclude_length))
            ]
        
        filtered_len = len(df_no_missing)
        log_messages.append(f"\nRemoved {original_len - filtered_len} rows containing excluded phrases (length < {cfg.data.exclude_length})")
        
        # Save excluded rows
        if not excluded_rows.empty:
            excluded_rows.to_csv(data_dir / 'excluded_phrases_data.csv', index=True)
            log_messages.append(f"Saved {len(excluded_rows)} rows containing excluded phrases")
    
    # Duplicate data processing
    duplicates_all = df_no_missing[df_no_missing.duplicated(subset=['text'], keep=False)].copy()
    if not duplicates_all.empty:
        # Add group ID to duplicate data
        duplicates_all['duplicate_group'] = duplicates_all.groupby('text').ngroup()
        
        # Only first item in each group is kept, others are marked for removal
        duplicates_all['status'] = 'removed'
        duplicates_all.loc[~duplicates_all.duplicated(subset=['text'], keep='first'), 'status'] = 'kept'
        
        # Sort by duplicate_group and then by status (kept comes first)
        duplicates_all = duplicates_all.sort_values(['duplicate_group', 'status'], 
                                                  ascending=[True, True])
        
        # Save duplicate data (both kept and removed)
        duplicates_all.to_csv(data_dir / 'duplicate_data.csv', index=True)
        log_messages.append(f"\nSaved {len(duplicates_all)} rows of duplicate data (including kept and removed)")
        log_messages.append(f"- Kept: {len(duplicates_all[duplicates_all['status'] == 'kept'])} rows")
        log_messages.append(f"- Removed: {len(duplicates_all[duplicates_all['status'] == 'removed'])} rows")
    
    # Remove duplicates (keep first item)
    df_final = df_no_missing.drop_duplicates(subset=['text'], keep='first')
    
    # Calculate emotion class statistics
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
    
    # Save removed data statistics
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
    
    with open(data_dir / 'removal_stats.json', 'w', encoding='utf-8') as f:
        json.dump(removal_stats, f, indent=2, ensure_ascii=False)
    
    log_messages.append("\nData preprocessing stats after:")
    log_messages.append(f"Total samples: {len(df_final)}")
    log_messages.append(f"Total removed samples:")
    log_messages.append(f"- Missing values: {len(missing_data)}")
    log_messages.append(f"- Excluded phrases: {len(excluded_rows)}")
    log_messages.append(f"- Duplicates: {len(duplicates_all[duplicates_all['status'] == 'removed']) if not duplicates_all.empty else 0}")
    log_messages.append("\nEmotion class statistics:")
    for emotion, stats in emotion_stats.items():
        log_messages.append(f"\n{emotion}:")
        log_messages.append(f"- Initial count: {stats['initial_count']}")
        log_messages.append(f"- Excluded: {stats['excluded_count']}")
        log_messages.append(f"- Duplicates removed: {stats['duplicates_removed']}")
        log_messages.append(f"- Final count: {stats['final_count']}")
        log_messages.append(f"- Removal percentage: {stats['removal_percentage']:.2f}%")
    
    # Combine log messages into a single string
    log_message = "\n".join(log_messages)
    
    # Log to file
    if logger:
        safe_log(logger, 'info', log_message)
    else:
        print(log_message)
    
    return df_final, labels

def clean_json_response(response_text: str) -> str:
    """Clean JSON response"""
    try:
        print("\nRaw response:", repr(response_text))
        
        # 응답이 단순 문자열인 경우 처리
        if isinstance(response_text, str):
            # "Expected Output:" 텍스트 제거
            response_text = re.sub(r'^.*?Expected Output:\s*', '', response_text, flags=re.IGNORECASE)
            
            # JSON 블록 찾기 (중첩된 중괄호도 처리)
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if matches:
                json_text = matches[-1]  # 마지막 JSON 객체 사용
                print("Extracted JSON:", repr(json_text))
                
                # JSON 파싱 시도
                try:
                    parsed = json.loads(json_text)
                    if isinstance(parsed, dict) and "emotion" in parsed:
                        return json.dumps(parsed, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            
            # 감정 단어 추출 시도
            emotion_pattern = r'emotion:\s*(\w+)'
            emotion_match = re.search(emotion_pattern, response_text, re.IGNORECASE)
            
            if emotion_match:
                emotion = emotion_match.group(1).lower()
                return json.dumps({
                    "emotion": emotion,
                    "confidence_score": 0.5,
                    "explanation": "Emotion extracted from text response"
                }, ensure_ascii=False)
            
            # 응답에서 알파벳 문자열만 추출
            word_match = re.search(r'[a-zA-Z]+', response_text)
            if word_match:
                emotion_word = word_match.group(0).lower()
                return json.dumps({
                    "emotion": emotion_word,
                    "confidence_score": 0.5,
                    "explanation": "Extracted from non-JSON response"
                }, ensure_ascii=False)
        
        # 모든 처리 실패 시 기본값 반환
        return json.dumps({
            "emotion": "unknown",
            "confidence_score": 0.0,
            "explanation": "Failed to parse response"
        }, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error in clean_json_response: {e}")
        print(f"Original response: {response_text}")
        return json.dumps({
            "emotion": "unknown",
            "confidence_score": 0.0,
            "explanation": f"Error processing response: {str(e)}"
        }, ensure_ascii=False)


def get_model_response(text: str, labels: list, client, model: str, temperature: float, cfg, logger=None, rag=None, tools=None, count=0) -> dict:
    """Get model response with RAG or template"""
    global _PROMPT_LOGGED
  
    final_prompt = ""  # 초기화 추가
    template = ""      # 초기화 추가
    
    try:
        # RAG 프롬프트 처리
        # 템플릿 프롬프트 처리
     
        try:
            template_name = cfg.model.template
            
            if logger:
                logger.debug("Using template: {}".format(template_name))

            if template_name == "rag_prompt":
                try:
                    if rag is None:
                        raise ValueError("RAG is enabled but rag object is None")
                    similar_examples = rag.get_similar_examples(
                        text, 
                        k=cfg.rag.k_examples, 
                        threshold=cfg.rag.threshold
                    )
                    final_prompt = str(rag.get_rag_prompt(text, similar_examples))
                    if logger:
                        logger.debug(f"Using RAG prompt with {len(similar_examples)} examples (threshold: {cfg.rag.threshold})")
                except Exception as e:
                    if logger:
                        logger.error("Error in RAG processing: {}".format(str(e)))
            else:
                template = getattr(cfg.prompt, template_name)
                final_prompt = template
        except Exception as e:
            if logger:
                logger.error("Error in template processing: {}".format(str(e)))
            # 템플릿 실패 시 기본 프롬프트로 폴백
            # template = "basic_prompt"
            # final_prompt = "Classify the emotion in this text as one of: {}\n\nText: {}".format(', '.join(labels), text)

        
        # 최종 프롬프트 구성
        final_prompt = "{}\n\nText: {}".format(final_prompt, text)
        
        # 로깅 처리
        if logger and count == 0:
            try:
                safe_prompt = final_prompt.encode('ascii', errors='replace').decode('ascii')
                logger.debug("Using prompt: \n{}".format(safe_prompt))
            
            except Exception as e:
                logger.error("Error logging prompt: {}".format(str(e)))

        # 모델 응답 처리
        try:
            if cfg.model.type == "ollama":
                response = client.invoke(final_prompt)
            elif cfg.model.type == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=cfg.model.max_tokens,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=temperature
                )
                response = response.content[0].text
            else:
                messages = [{"role": "system", "content": final_prompt}]
                use_function_call = any(model_name.lower() in model.lower() 
                                      for model_name in cfg.prompt.function_call_models)
                
                if use_function_call and cfg.model.function_calling:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        tools=tools,
                        seed=int(cfg.general.seed)
                    )
                    if response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]
                        result = json.loads(tool_call.function.arguments)
                        if logger:
                            try:
                                logger.debug("Function call result: {}".format(
                                    json.dumps(result, ensure_ascii=True, indent=2)
                                ))
                            except Exception as e:
                                logger.error("Error logging function call result: {}".format(str(e)))
                        return result, final_prompt
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
                    response = response.choices[0].message.content

            # 응답 파싱 및 검증
            if logger:
                try:
                    safe_response = str(response).encode('ascii', errors='replace').decode('ascii')
                    logger.debug("Raw response: \n{}".format(safe_response))
                except Exception as e:
                    logger.error("Error logging response: {}".format(str(e)))
                    
            result = clean_json_response(response)
            if isinstance(result, str):
                result = json.loads(result)

            # 결과 검증
            required_fields = ["emotion", "confidence_score", "explanation"]
            if not all(field in result for field in required_fields):
                if logger:
                    logger.warning("Missing required fields in response, retrying with structured prompt...")
                return retry_emotion_prediction(text, labels, client, model, temperature, int(cfg.general.seed), cfg), final_prompt

            if result["emotion"] not in labels:
                if logger:
                    logger.warning("Invalid emotion label '{}', retrying with structured prompt...".format(
                        str(result["emotion"]).encode('ascii', errors='replace').decode('ascii')
                    ))
                return retry_emotion_prediction(text, labels, client, model, temperature, int(cfg.general.seed), cfg), final_prompt

            return result, final_prompt

        except json.JSONDecodeError:
            if logger:
                logger.warning("JSON parsing failed, retrying with structured prompt...")
            return retry_emotion_prediction(text, labels, client, model, temperature, int(cfg.general.seed), cfg), final_prompt

        except Exception as e:
            if logger:
                logger.error("Error generating response: {}".format(str(e)))
            raise

    except Exception as e:
        if logger:
            logger.error("Error in get_model_response: {}".format(str(e)))
        return retry_emotion_prediction(text, labels, client, model, temperature, int(cfg.general.seed), cfg), final_prompt

class SafeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.encoding = 'utf-8'
        if sys.platform == 'win32':
            if stream is None:
                self.stream = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
            elif hasattr(stream, 'buffer'):
                self.stream = codecs.getwriter('utf-8')(stream.buffer, errors='replace')

    def format(self, record):
        """Format the specified record safely."""
        try:
            message = super().format(record)
            if sys.platform == 'win32':
                # Windows에서는 ASCII로 변환
                return message.encode('ascii', errors='replace').decode('ascii')
            return message
        except Exception as e:
            return f"Error formatting log message: {str(e)}"

    def emit(self, record):
        """Emit a record safely."""
        try:
            msg = self.format(record)
            stream = self.stream
            
            try:
                if isinstance(msg, str):
                    stream.write(msg + self.terminator)
                else:
                    stream.write(str(msg) + self.terminator)
                self.flush()
            except (UnicodeEncodeError, UnicodeDecodeError):
                # 인코딩 에러 발생 시 ASCII로 변환 시도
                try:
                    safe_msg = str(msg).encode('ascii', errors='replace').decode('ascii')
                    stream.write(safe_msg + self.terminator)
                    self.flush()
                except Exception as e:
                    sys.stderr.write(f"Severe logging error: {str(e)}\n")
                    self.handleError(record)
            except Exception as e:
                sys.stderr.write(f"Logging error: {str(e)}\n")
                self.handleError(record)
        except Exception:
            self.handleError(record)

def setup_logging(cfg, output_dir: Path):
    """Logging setup"""
    # Create log directory
    log_dir = output_dir / cfg.general.logging.log_path
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log file path
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Logger setup
    logger = logging.getLogger('emotion_analysis')
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Windows environment setup for UTF-8 output
    if sys.platform == 'win32':
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        # File handler setup
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)

        # Console handler setup
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter setup
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Default console logger setup
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def safe_log(logger, level: str, message: str):
    """Safe logging function"""
    try:
        # Convert non-string types to string
        if not isinstance(message, str):
            message = str(message)
        
        # Log message based on level
        log_func = getattr(logger, level)
        log_func(message)
            
    except Exception as e:
        # Log failure fallback to default output
        print(f"Logging failed: {str(e)}")
        print(f"Original message: {message}")

def get_output_dir_name(model_name: str, cfg) -> str:
    """Create output folder name based on settings"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [timestamp, model_name]
    
 
    template_name = str(cfg.model.template)
    if isinstance(template_name, str) and template_name.startswith("${prompt."):
        # Extract the actual template name from the reference
        template_name = template_name.split("${prompt.")[1].rstrip("}")
    name_parts.append(template_name)
    
    # # Add RAG if enabled
    # if cfg.model.use_rag:
    #     name_parts.append("rag")
    
    # # Add function calling if enabled
    # if cfg.model.function_calling:
    #     name_parts.append("function_calling")
    
    return "_".join(name_parts)

def initialize_models(cfg):
    """Initialize LLM and RAG model"""
    client = None
    
    if cfg.model.type == "ollama":
        client = OllamaLLM(
            model=cfg.model.name,
            temperature=cfg.model.temperature
        )
    elif cfg.model.type == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif cfg.model.type == "upstage":
        base_url = "https://api.upstage.ai/v1/solar"
        api_key = os.getenv("UPSTAGE_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif cfg.model.type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        client = Anthropic(api_key=api_key)
    

    return client

def sanitize_model_name(model_name: str) -> str:
    """Clean model name for use as file name"""
    return re.sub(r'[^\w\-\.]', '_', model_name.lower())

@hydra.main(version_base="1.2", config_path='config', config_name='llm')
def main(cfg):
    # 전역 변수 사용 선언
    global _INPUT_LOGGED
    
    # Initialize models
    client = initialize_models(cfg)
    
    # Convert config values to Python basic types first
    model = str(cfg.model.name)
    temperature = float(cfg.model.temperature)
    
    # Set up output directory first
    model_name = sanitize_model_name(model)
    
    output_dir_name = get_output_dir_name(model_name, cfg)
    output_dir = Path('outputs') / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update cfg output_path
    cfg.general.output_path = str(output_dir)
    
    # Now setup logging with the created output directory
    logger = setup_logging(cfg, output_dir)
    
    # 데이터 로딩 및 전처리
    df_isear, labels = load_data(cfg, logger)
    labels = list(map(str, labels))
    
    # Set n_samples
    n_samples = len(df_isear) if cfg.data.n_samples == -1 else min(cfg.data.n_samples, len(df_isear))
    logger.info(f"Using sanitized model name: {model_name}")
    logger.info(f"Processing {n_samples} samples out of {len(df_isear)} total samples")
    
    # Select first n_samples
    df_isear = df_isear.head(n_samples)
    
    # Set output file path (directory already created)
    output_path = Path(cfg.general.output_path) / f'dataset-{cfg.data.name}_model-{model_name}.csv'
    
    # Function calling tool definition
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
                            "enum": list(labels),
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

    # Initialize invalid prediction statistics
    invalid_predictions = {
        "total": 0,
        "invalid_samples": [],
        "by_true_label": {str(label): 0 for label in labels}
    }

    # Initialize RAG
    rag = None
    try:
        if cfg.model.template == "rag_prompt":
            logger.info("Initializing RAG...")
            try:
                rag = EmotionRAG(cfg)
                rag.create_index(df_isear)
                logger.info("RAG initialization completed")
            except Exception as e:
                logger.error(f"Error initializing RAG: {str(e)}")
                logger.warning("Continuing without RAG...")
    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}")
        logger.warning("Continuing without RAG...")

    # 데이터 정제 후에 tqdm을 사용하도록 수정
    total_samples = len(df_isear)  # 실제 처리할 샘플 수
    
    # 로깅을 위한 구분자 정의
    log_separator = "="*80
    
    for index, row in tqdm(df_isear.iterrows(), total=total_samples):
        try:
            if rag:
                rag.exclude_index(index)
                
            result, _= get_model_response(
                str(row.text),
                labels,
                client,
                model,
                temperature,
                cfg,
                logger=logger,
                rag=rag,
                tools=tools,
                count=index
            )
            
            # 예측 결과 저장
            df_isear.at[index, f'predicted_emotion_{model_name}'] = result["emotion"]
            df_isear.at[index, f'confidence_score_{model_name}'] = result["confidence_score"]
            df_isear.at[index, f'explanation_{model_name}'] = result["explanation"]
            
            # 예측 결과 로깅 (매 샘플마다)
            result_log = f"""
{log_separator}
[Sample Index: {index}] Prediction Result
{log_separator}
Input Text: {str(row.text).encode('ascii', errors='ignore').decode('ascii')}
Ground Truth: {str(row.emotion).encode('ascii', errors='ignore').decode('ascii')}
Predicted Emotion: {str(result['emotion']).encode('ascii', errors='ignore').decode('ascii')}
Confidence Score: {float(result['confidence_score'])}
Explanation: {str(result.get('explanation', '')).encode('ascii', errors='ignore').decode('ascii')}
{log_separator}
"""
            safe_log(logger, 'info', result_log)
            
            # 프롬프트는 처음 한 번만 로깅
            if not _INPUT_LOGGED:
                input_log = f"""
{log_separator}
[Initial Sample] Prompt Details

"""
                safe_log(logger, 'debug', input_log)
                _INPUT_LOGGED = True

            if index % 500 == 0:
                df_isear.to_csv(output_path, index=False)

                

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            # 에러 발생 시에만 unknown으로 설정
            df_isear.at[index, f'predicted_emotion_{model_name}'] = "unknown"
            df_isear.at[index, f'confidence_score_{model_name}'] = 0.0
            df_isear.at[index, f'explanation_{model_name}'] = f"Error: {str(e)}"
            
            # 에러 로깅
            error_log = f"""
{log_separator}
[Error at Sample Index: {index}]
{log_separator}
Input Text: {str(row.text).encode('ascii', errors='ignore').decode('ascii')}
Ground Truth: {str(row.emotion).encode('ascii', errors='ignore').decode('ascii')}
Error: {str(e)}
{log_separator}
"""
            safe_log(logger, 'error', error_log)
            continue

    # Final result save
    df_isear.to_csv(output_path, index=False)
    
    # Save invalid prediction statistics
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
    
    # Save statistics
    save_prediction_stats(df_isear, invalid_stats, output_dir, model_name, labels)
    
    # Calculate and save evaluation metrics
    report, cm = save_metrics(df_isear, cfg, model_name, output_dir)
    
    # Final results log creation
    final_log = f"""
{'='*80}
[Final Results Summary]
{'='*80}

1. Wrong Prediction Statistics:
- Total samples: {invalid_stats['total_samples']}
- Wrong predictions: {invalid_stats['invalid_count']}
- Wrong prediction percentage: {invalid_stats['invalid_percentage']:.2f}%

2. Classification Performance:
- Overall accuracy: {report['overall']['accuracy']:.4f}

3. Class-wise Performance:"""

    # Add per-class performance metrics
    for label, metrics in report['per_class'].items():
        final_log += f"""
{label}:
  - Precision: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - F1 Score: {metrics['f1-score']:.4f}"""

    final_log += f"\n{'='*80}\n"
    
    # Log to file
    safe_log(logger, 'info', final_log)
    
    print("\nClassification report summary:")
    print(f"Accuracy: {report['overall']['accuracy']:.4f}")
    print("\nClass-wise metrics:")
    for label, metrics in report['per_class'].items():
        print(f"{label}:")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall: {metrics['recall']:.4f}")
        print(f"  f1-score: {metrics['f1-score']:.4f}")

def save_prediction_stats(df, invalid_stats, output_dir, model_name, labels):
    """Save prediction statistics"""
    try:
        # Save statistics file
        stats_file = output_dir / f'prediction_stats_{model_name}.json'
        with open(stats_file, 'w', encoding='ascii') as f:
            json.dump(invalid_stats, f, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"Error saving prediction stats: {e}")

def metric():
    cfg = OmegaConf.load("config/llm.yaml")
    
    # Convert path to Path object for correct path construction
    base_path = Path("D:/dev/isp_rag_cluster/outputs")
    
    # Verify actual existence of directory
    output_folders = [f for f in base_path.iterdir() if f.is_dir()]
    if not output_folders:
        raise FileNotFoundError("No output folders found")
    
    # Select latest folder (based on timestamp)
    latest_folder = max(output_folders, key=lambda x: x.stat().st_mtime)
    latest_folder = Path(r"outputs\20250303_172613_qwen2.5_emotion_prompt\dataset-isear_model-qwen2.5")
    # File path construction
    model_name = latest_folder.name.split("_model-")[1]
    #model_name = "llama3.2"  # Or other model name
    
    file_name = f"dataset-isear_model-{model_name}.csv"
    latest_folder = latest_folder.parent
    file_path = latest_folder / file_name
    
    print(f"Looking for file at: {file_path}")  # For debugging
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read file
    df_isear = pd.read_csv(file_path)
    
    # Save metrics
    save_metrics(df_isear, cfg, model_name, latest_folder)

if __name__ == "__main__":
    main()
    #metric()
    
