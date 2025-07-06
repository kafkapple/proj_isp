import json
import pandas as pd
from pathlib import Path
import re
from src.logger import safe_log
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

