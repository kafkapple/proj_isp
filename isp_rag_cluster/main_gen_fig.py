from src.gen_table import gen_table
from src.gen_plot import plot_grouped_bar_chart

import os
import shutil
import pandas as pd
from pathlib import Path
# 기본 경로 설정 (필요에 따라 수정)
path_bar = 'data/preped_results'    # 최종 for_bar.csv 저장 경로
path_table = 'data/preped_results/for_table'  # 변경된 classification_report 파일 저장 경로

# 출력 경로가 없으면 생성
os.makedirs(path_bar, exist_ok=True)
os.makedirs(path_table, exist_ok=True)

#

def prep_results(base_dir, prompt_types):
    # 결과를 저장할 리스트 초기화
    results = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            # 폴더명이 "date_time_나머지" 형태여야 함 (최소 3부분)
            parts = folder.split('_', 2)  # date, time, 나머지
            if len(parts) < 3:
                continue
            rest = parts[2]  # 예: "qwen2.5_3b_rag_prompt"
            print(f"Processing folder: {rest}")
            
            model_name = None
            prompt_type_found = None
            
            # 프롬프트 타입 리스트에서 가장 긴 매칭을 찾음
            matched_length = -1  # 매칭된 문자열의 길이를 저장
            for pt in prompt_types:
                search_str = '_' + pt
                if search_str in rest:
                    idx = rest.rfind(search_str)
                    current_length = len(pt)
                    if current_length > matched_length:
                        matched_length = current_length
                        model_name = rest[:idx]
                        prompt_type_found = pt
                        print(f"Found match: {pt} (length: {current_length})")
            
            if model_name is None or prompt_type_found is None:
                print(f"Skipping folder - no matching prompt type: {rest}")
                continue
                
            print(f"Final parse result: model_name: {model_name}, prompt_type: {prompt_type_found}")

            # 폴더 내에서 'classification_report_'로 시작하고 .csv로 끝나는 파일 찾기
            for file in os.listdir(folder_path):
                if file.startswith("classification_report_") and file.endswith(".csv"):
                    src_file = os.path.join(folder_path, file)
                    # 새 파일명: classification_report_{model_name}_{prompt_type}.csv
                    new_filename = f"classification_report_{model_name}_{prompt_type_found}.csv"
                    dest_file = os.path.join(path_table, new_filename)
                    shutil.copy(src_file, dest_file)
                    
                    # CSV 파일 읽기 및 macro avg 행의 precision, recall, f1-score 추출
                    try:
                        print(f"\nProcessing file: {src_file}")
                        # 먼저 인덱스 없이 읽어보기
                        df = pd.read_csv(src_file)
                        
                        # 'class' 또는 첫 번째 컬럼이 인덱스인지 확인
                        if 'class' in df.columns:
                            macro_row = df[df['class'] == 'macro avg']
                        else:
                            # 첫 번째 컬럼을 인덱스로 설정
                            df.set_index(df.columns[0], inplace=True)
                            macro_row = df.loc['macro avg'] if 'macro avg' in df.index else None
                        
                        if macro_row is not None:
                            if isinstance(macro_row, pd.Series):
                                precision = macro_row.get('precision', None)
                                recall = macro_row.get('recall', None)
                                f1_score = macro_row.get('f1-score', None)
                            else:
                                precision = macro_row['precision'].values[0]
                                recall = macro_row['recall'].values[0]
                                f1_score = macro_row['f1-score'].values[0]
                            
                            result = {
                                'model name': model_name,
                                'template type': prompt_type_found,
                                'precision': precision,
                                'recall': recall,
                                'f1-score': f1_score
                            }
                            print(f"Added result: {result}")
                            results.append(result)
                        else:
                            print(f"Warning: 'macro avg' row not found in {src_file}")
                            
                    except Exception as e:
                        print(f"Error processing file ({src_file}): {str(e)}")
                        print("DataFrame head:")
                        try:
                            print(df.head())
                        except:
                            pass
                    break  # 해당 폴더는 한 파일만 처리

    # 결과 확인 및 저장
    if results:
        print("\nAll collected results:")
        for r in results:
            print(f"Model: {r['model name']}, Template: {r['template type']}")
            
        df_bar = pd.DataFrame(results)
        output_path = Path(path_bar) / "for_bar.csv"
        df_bar.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")
        print("DataFrame contents:")
        print(df_bar)


if __name__ == "__main__":
    path_target = "./outputs/done_4"
    # 프롬프트 타입 리스트 (예시: 'rag_prompt' 외 필요시 추가)
    prompt_order = ['baseline_prompt', 'zero_shot_prompt', 'few_shot_prompt',  'custom_prompt',  'rag_prompt', 'rag_prompt-2-savani' ]
    model_order = [ 'llama3.2', 'llama3.1','llama2_13b','qwen2.5_3b', 'qwen2.5', 'qwen2.5_14b', 'mistral', 'gemma','gpt-4o',  'claude-3-5-sonnet-20241022']
    name_mapping = {
    'qwen2.5_3b': 'Qwen 2.5 3B',
    'qwen2.5': 'Qwen 2.5 7B',
    'qwen2.5_14b': 'Qwen 2.5 14B',
    'llama3.2': 'Llama 3.2 3B',
    'llama3.1': 'Llama 3.1 7B',
    'llama2_13b': 'Llama 2 13B',
    'mistral': 'Mistral 7B',
    'gemma3': 'Gemma3 4B',
    'gemma': 'Gemma 7B',
    'gpt-4o': 'GPT-4o',
    'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
    'baseline_prompt': 'Baseline',
    'zero_shot_prompt': 'Zero-shot',
    'few_shot_prompt': 'Few-shot',
    'custom_prompt': 'Custom',
    'rag_prompt': 'RAG',
    'rag_prompt-2-savani': 'RAG-Distilbert',
    }

    prep_results(base_dir = path_target, prompt_types = prompt_order)
    gen_table(csv_dir = path_table,  output_name = f"{path_target}/classification_table_new", round_digits = 0, name_mapping = name_mapping, prompt_order = prompt_order)
    # 사용 예시
    plot_grouped_bar_chart(Path(path_bar) / "for_bar.csv", f"{path_target}/grouped_bar_chart_new", name_mapping = name_mapping, model_order = model_order, prompt_order = prompt_order)