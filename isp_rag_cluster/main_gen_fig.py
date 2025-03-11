
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

# 프롬프트 타입 리스트 (예시: 'rag_prompt' 외 필요시 추가)
prompt_types = ['baseline_prompt', 'zero_shot_prompt', 'few_shot_prompt',  'custom_prompt',  'rag_prompt' ]

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
            print(f"rest: {rest}")
            
            model_name = None
            prompt_type_found = None
            
            # 프롬프트 타입 리스트에 있는 값 중 폴더명에 포함된 것을 찾음
            for pt in prompt_types:
                search_str = '_' + pt  # 앞에 '_' 포함하여 검색
                if search_str in rest:
                    idx = rest.rfind(search_str)  # 마지막 등장 위치 (모델명에 '_'가 있을 수 있으므로)
                    model_name = rest[:idx]  # _pt 이전까지가 모델명
                    prompt_type_found = pt
                    print(f"model_name: {model_name}, prompt_type_found: {prompt_type_found}")
                    break
            # 파싱 실패 시 건너뛰기
            if model_name is None or prompt_type_found is None:
                continue

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
                        # CSV 파일 구조에 따라 인덱스를 지정 (일반적으로 sklearn classification_report는 index에 label 포함)
                        df = pd.read_csv(src_file, index_col=0)
                        if 'macro avg' in df.index:
                            row = df.loc['macro avg']
                            precision = row.get('precision', None)
                            recall = row.get('recall', None)
                            f1_score = row.get('f1-score', None)
                            
                            results.append({
                                'model name': model_name,
                                'template type': prompt_type_found,
                                'precision': precision,
                                'recall': recall,
                                'f1-score': f1_score
                            })
                    except Exception as e:
                        print(f"파일 읽기 에러 ({src_file}): {e}")
                    break  # 해당 폴더는 한 파일만 처리한다고 가정

    # 모든 결과를 DataFrame으로 변환하여 for_bar.csv로 저장
    if results:
        df_bar = pd.DataFrame(results)
        output_path = Path(path_bar) / "for_bar.csv"
        df_bar.to_csv(output_path, index=False)


if __name__ == "__main__":
    path_target = "./outputs/done_4"
    model_order = [ 'llama3.2', 'llama3.1','qwen2.5_3b', 'qwen2.5', 'mistral']
    name_mapping = {
    'qwen2.5_3b': 'Qwen 2.5 3B',
    'qwen2.5': 'Qwen 2.5 7B',
    'llama3.1': 'Llama 3.1 7B',
    'llama3.2': 'Llama 3.2 3B',
    'mistral': 'Mistral 7B',
    'baseline_prompt': 'Baseline',
    'zero_shot_prompt': 'Zero-shot',
    'few_shot_prompt': 'Few-shot',
    'custom_prompt': 'Custom',
    'rag_prompt': 'RAG'
    }

    prep_results(base_dir = path_target, prompt_types = prompt_types)
    gen_table(csv_dir = path_table,  output_name = f"{path_target}/classification_table_new", round_digits = 0, name_mapping = name_mapping)
    # 사용 예시
    plot_grouped_bar_chart(Path(path_bar) / "for_bar.csv", f"{path_target}/grouped_bar_chart_new", name_mapping = name_mapping, model_order = model_order)