from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import json
import re
from ollama import chat, ChatResponse
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

def save_metrics(df, cfg, model, output_dir):
    # 데이터 전처리
    print("\nData preprocessing stats before:")
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # 감정 라벨 소문자로 변환
    df['emotion'] = df['emotion'].str.lower()
    df[f'predicted_emotion_{model}'] = df[f'predicted_emotion_{model}'].str.lower()
    
    # 결측치와 중복 제거
    df = df.dropna(subset=['emotion', f'predicted_emotion_{model}'])
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
    y_pred = df[f'predicted_emotion_{model}'].fillna(default_emotion).astype(str)
    
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
    with open(output_dir / f'classification_report_{model}.txt', "w") as file:
        file.write(report_str)
    
    # Classification Report를 DataFrame으로 변환하여 CSV로 저장
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / f'classification_report_{model}.csv')
    
    # 혼동 행렬 계산 (명시적으로 라벨 지정)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    # 결과 저장을 위한 경로 설정
    
    metrics_path = output_dir / f'metrics_{model}.json'
    
    # 메트릭 결과 저장
    metrics_result = {
        "overall": {
            "accuracy": report_dict['accuracy'],
            "macro_avg": report_dict['macro avg'],
            "weighted_avg": report_dict['weighted avg']
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
    ax1.set_title(f'Confusion Matrix - {model}')
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
    ax2.set_title(f'Normalized Confusion Matrix - {model}')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 두 subplot 모두에 대해 라벨 회전 및 정렬 조정
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # 그래프가 잘리지 않도록 레이아웃 조정
    plt.tight_layout()
    
    # 더 높은 해상도로 저장
    plt.savefig(output_dir / f'confusion_matrix_{model}.png', 
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
        #  Column names contain the separator
        df.columns = [col.split(dataset_cfg.separator)[0] for col in df.columns]
    
    # Select only the required columns
    df = df[required_columns]
    df.rename(columns={'SIT': 'text', 'EMOT': 'emotion'}, inplace=True)
    df['emotion'] = df['emotion'].map(lambda x: labels[int(x)-1] if x.isdigit() else 'undefined')
    
    # Normalize emotion labels
    print(f"Original emotion labels: {df['emotion'].unique()}")

    df['emotion'] = df['emotion'].str.lower()
    print(f"Normalized emotion labels: {df['emotion'].unique()}")
    # Drop missing values and duplicates
    print(f"Missing values: {df.isna().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    duplicated_rows = df[df.duplicated(keep=False)]
    print(duplicated_rows)  # print duplicated rows

    df = df.dropna()
    df = df.drop_duplicates()
    return df, labels

@hydra.main(version_base="1.2", config_path='config', config_name='llm')
def main(cfg):
    # 모든 config 값들을 Python 기본 타입으로 변환
    model_type = str(cfg.model.type)
    model = str(cfg.model.name)
    temperature = float(cfg.model.temperature)
    prompt = str(cfg.prompt.qwen)
    
    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs') / f"{timestamp}_{model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # cfg의 output_path 업데이트
    cfg.general.output_path = str(output_dir)
    
    
    # 데이터 로드 및 라벨 준비
    df_isear, labels = load_data(cfg)
    labels = list(map(str, labels))
    
    # n_samples 설정
    n_samples = len(df_isear) if cfg.data.n_samples == -1 else cfg.data.n_samples
    print(f"Processing {n_samples} samples out of {len(df_isear)} total samples")
    
    # 출력 파일 경로 설정 (디렉토리는 이미 생성됨)
    output_path = Path(cfg.general.output_path) / f'dataset-{cfg.data.name}_model-{model}.csv'

    if model_type == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        llm_model = "gpt-3.5-turbo-1106"
    elif model_type == "upstage":
        base_url = "https://api.upstage.ai/v1/solar"
        api_key = os.getenv("UPSTAGE_API_KEY")
        llm_model = "solar-1-mini-chat"
    elif model_type == "ollama":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"

    client = OpenAI(
                base_url=base_url,
                api_key=api_key  
            )
    # Function calling 도구 정의 - 모든 값을 기본 Python 타입으로 확실하게 변환
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

    # 잘못된 감정 예측 통계
    invalid_predictions = {
        "total": 0,
        "samples": [],
        "by_true_label": {str(label): 0 for label in labels}  # 문자열로 변환
    }

    for index, row in tqdm(df_isear.iterrows(), total=n_samples):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": str(row.text)}],
                tools=tools,
                seed=int(cfg.general.seed),
                temperature=temperature,
                timeout=10
            )

            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                # 더 안전한 값 추출 및 타입 변환
                try:
                    emotion = str(function_args.get("emotion", "unknown")).lower()
                    print(f"Initial emotion: {emotion}")  # 초기 감정 출력
                except:
                    emotion = "unknown"
                    print("Failed to get emotion, using 'unknown'")
                    
                try:
                    confidence_score = float(function_args.get("confidence_score", 0.0))
                    if confidence_score is None:
                        confidence_score = 0.0
                except (TypeError, ValueError):
                    confidence_score = 0.0
                    
                try:
                    explanation = str(function_args.get("explanation", "No explanation provided"))
                except:
                    explanation = "No explanation provided"
                
                # 예측된 감정이 유효한지 확인
                if emotion not in labels:
                    print(f"Invalid emotion detected: {emotion}")  # 유효하지 않은 감정 출력
                    invalid_predictions["total"] += 1
                    invalid_predictions["by_true_label"][str(row.emotion)] += 1
                    invalid_predictions["samples"].append({
                        "index": int(index),
                        "text": str(row.text),
                        "true_emotion": str(row.emotion),
                        "predicted_emotion": emotion,
                        "confidence": confidence_score
                    })
                    
                    # 설정 존재 여부 체크 후 재시도 여부 결정
                    should_retry = getattr(cfg.general, 'retry_invalid_predictions', False)
                    
                    if should_retry:
                        print("Attempting retry...")  # 재시도 시작
                        try:
                            retry_result = retry_emotion_prediction(
                                str(row.text), labels, client, model, temperature, int(cfg.general.seed)
                            )
                            print(f"Retry result: {retry_result}")  # 재시도 결과 출력
                            emotion = str(retry_result.get("emotion", "unknown"))
                            confidence_score = float(retry_result.get("confidence_score", 0.0))
                            explanation = str(retry_result.get("explanation", "No explanation from retry"))
                            
                            # 여전히 잘못된 경우 매핑
                            if emotion not in labels:
                                print(f"Retry emotion still invalid: {emotion}, applying mapping")
                                emotion = map_unknown_emotion(emotion, labels)
                        except Exception as e:
                            print(f"Error in retry for row {index}: {e}")
                            emotion = map_unknown_emotion(emotion, labels)
                    else:
                        print("Applying direct mapping")
                        emotion = map_unknown_emotion(emotion, labels)

            # 결과 저장 전 최종 타입 체크 및 출력
            emotion = str(emotion) if emotion else "unknown"
            confidence_score = float(confidence_score) if confidence_score is not None else 0.0
            explanation = str(explanation) if explanation else "No explanation"
            
            print(f"\nFinal result for row {index}:")
            print(f"Text: {row.text}")
            print(f"Ground truth: {row.emotion}")
            print(f"Predicted emotion: {emotion}")
            print(f"Confidence score: {confidence_score}")
            print(f"Explanation: {explanation}")
            print("-" * 80)

            # 결과 저장
            df_isear.at[index, f'predicted_emotion_{model}'] = emotion
            df_isear.at[index, f'confidence_score_{model}'] = confidence_score
            df_isear.at[index, f'explanation_{model}'] = explanation

            if index % 500 == 0:
                df_isear.to_csv(output_path, index=False)
            
            if index >= n_samples - 1:
                break

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            # 오류 발생 시 기본값 저장
            df_isear.at[index, f'predicted_emotion_{model}'] = "unknown"
            df_isear.at[index, f'confidence_score_{model}'] = 0.0
            df_isear.at[index, f'explanation_{model}'] = f"Error: {str(e)}"
            continue

    # 최종 결과 저장
    df_isear.to_csv(output_path, index=False)
    
    # 잘못된 예측 통계 저장
    invalid_stats = {
        "total_samples": n_samples,  # 실제 처리된 샘플 수 사용
        "invalid_count": int(invalid_predictions["total"]),
        "invalid_percentage": float((invalid_predictions["total"] / n_samples) * 100),
        "by_true_label": {
            str(label): {
                "count": int(invalid_predictions["by_true_label"][str(label)]),
                "percentage": float((invalid_predictions["by_true_label"][str(label)] / n_samples) * 100)
            }
            for label in labels
        },
        "invalid_samples": invalid_predictions["samples"]
    }
    
    # 통계 저장
    stats_path = Path(str(cfg.general.output_path)) / f'invalid_predictions_{model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(stats_path, 'w') as f:
        json.dump(invalid_stats, f, indent=2)
    
    print("\nInvalid Prediction Statistics:")
    print(f"Total invalid predictions: {invalid_stats['invalid_count']} ({invalid_stats['invalid_percentage']:.2f}%)")
    print("\nBy true label:")
    for label, stats in invalid_stats["by_true_label"].items():
        print(f"{label}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    # 평가 메트릭 계산 및 저장
    report, cm = save_metrics(df_isear, cfg, model, output_dir)
    
    # 주요 메트릭 출력
    print("\nClassification Report Summary:")
    print(f"Accuracy: {report['overall']['accuracy']:.4f}")
    print("\nPer-class metrics:")
    for label, metrics in report['per_class'].items():
        print(f"{label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")

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
    latest_folder = Path("outputs/20250228_000911_gpt/dataset-isear_model-gpt-3.5-turbo") #Path("outputs/20250228_000911/dataset-isear_model-gpt-3.5-turbo")
    # 파일 경로 구성
    model = "gpt-3.5-turbo"  # 또는 다른 모델명
    model = latest_folder.name.split("_model-")[1]
    file_name = f"dataset-isear_model-{model}.csv"
    latest_folder = latest_folder.parent
    file_path = latest_folder / file_name
    
    print(f"Looking for file at: {file_path}")  # 디버깅용
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # 파일 읽기
    df_isear = pd.read_csv(file_path)
    
    # 메트릭 저장
    save_metrics(df_isear, cfg, model, latest_folder)

if __name__ == "__main__":
    main()
    #metric()
    
