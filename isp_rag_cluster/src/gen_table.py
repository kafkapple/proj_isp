import os
import re
import pandas as pd
import numpy as np
import dataframe_image as dfi


def highlight_max(s, style='bold'):
    """Series에서 최대값에 대해 스타일 지정 (bold 또는 background-color)"""
    is_max = s == s.max()
    if style == 'bold':
        return ['font-weight: bold' if v else '' for v in is_max]
    else:  # background color
        return [f'background-color: yellow' if v else '' for v in is_max]

def format_number(value, round_digits=1):
    """숫자를 지정된 소수점 자리수로 반올림"""
    if pd.isna(value):
        return value
    return round(float(value), round_digits)

def gen_table(csv_dir = "./data/classification_reports", output_name = "classification_table", round_digits = 0, 
             name_mapping = None, model_order = None, prompt_order = None):
    """
    Parameters:
    -----------
    csv_dir : str
        분류 리포트 CSV 파일들이 있는 디렉토리 경로
    output_name : str
        출력할 테이블 이미지 파일 이름
    round_digits : int
        소수점 자리수
    name_mapping : dict, optional
        모델명과 프롬프트 타입을 변경하기 위한 매핑 딕셔너리
        예: {'qwen2.5_3b': 'Qwen 2.5', 'baseline_prompt': 'Baseline'}
    model_order : list, optional
        모델 순서를 지정하는 리스트 (매핑 이전의 원래 모델명 사용)
    prompt_order : list, optional
        프롬프트 타입 순서를 지정하는 리스트
    """
    # 폴더 내 CSV 파일 순회
    all_data = []
    original_models = []  # 원본 모델명 저장용
    
    for file_name in os.listdir(csv_dir):
        if file_name.endswith(".csv"):
            # 파일명에서 "classification_report_" 제거
            base_name = file_name.replace("classification_report_", "").replace(".csv", "")
            
            # prompt_types 리스트를 활용하여 파싱
            model = base_name
            prompt = "unknown"
            for pt in prompt_order:
                if f"_{pt}" in base_name:
                    parts = base_name.split(f"_{pt}")
                    model = parts[0]
                    prompt = pt
                    break
            
            # 원본 모델명 저장
            original_model = model
            
            # 매핑 적용
            if name_mapping:
                model = name_mapping.get(model, model)
                prompt = name_mapping.get(prompt, prompt)
            
            print(f"model: {model}, prompt: {prompt}")
            
            file_path = os.path.join(csv_dir, file_name)
            df = pd.read_csv(file_path)
            
            # 첫 번째 컬럼 이름이 비어있거나 Unnamed인 경우 'class'로 변경
            if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
                df.columns.values[0] = 'class'
            # 인덱스가 'metric'인 경우도 처리
            elif 'class' not in df.columns and df.index.name == 'metric':
                df = df.reset_index()
                df = df.rename(columns={'metric': 'class'})
            
            # sklearn.metrics.classification_report 형식으로 가정
            # 'class' 열에 'macro avg' 행을 찾음
            macro_row = df[df['class'] == 'macro avg']
            if macro_row.empty:
                continue
            # macro avg f1-score를 백분율로 변환
            macro_f1 = float(macro_row['f1-score'].values[0]) * 100
            
            # 'macro avg', 'weighted avg' 등 불필요한 행은 제외하고 실제 클래스만 선택
            class_rows = df[~df['class'].isin(['accuracy','micro avg', 'weighted avg'])]# #['macro avg', 'weighted avg'])]
            
            # 각 클래스별 metric 값 추출 (백분율)
            row_data = {('Model', ''): model, ('Prompt', ''): prompt, ('original_model', ''): original_model}
            for _, row in class_rows.iterrows():
                cls = row['class']
                try:
                    p = format_number(float(row['precision']) * 100, round_digits)
                    r = format_number(float(row['recall']) * 100, round_digits)
                    f = format_number(float(row['f1-score']) * 100, round_digits)
                except ValueError:
                    p = r = f = np.nan
                row_data[(cls, 'P')] = p
                row_data[(cls, 'R')] = r
                row_data[(cls, 'F1')] = f
            
            # Acc% (Macro F1)
            #row_data[('Acc%', '')] = format_number(macro_f1, round_digits)
            all_data.append(row_data)

    # 통합 DataFrame 구성 (MultiIndex 컬럼)
    final_df = pd.DataFrame(all_data)
    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)

    # 정렬을 위한 키 생성
    if model_order and prompt_order:
        # 모델 순서 매핑 (원본 모델명 기준)
        model_map = {model: idx for idx, model in enumerate(model_order)}
        final_df['model_sort'] = final_df[('original_model', '')].map(model_map)
        
        # 프롬프트 순서 매핑 (매핑된 이름 기준)
        prompt_map = {name_mapping.get(p, p) if name_mapping else p: idx 
                     for idx, p in enumerate(prompt_order)}
        final_df['prompt_sort'] = final_df[('Prompt', '')].map(prompt_map)
        
        # 모델과 프롬프트로 정렬
        final_df = final_df.sort_values(['model_sort', 'prompt_sort'])
        
        # 정렬에 사용한 컬럼 제거
        final_df = final_df.drop(['model_sort', 'prompt_sort'], axis=1)
        
        # 필터링: model_order에 있는 모델만 선택
        final_df = final_df[final_df[('original_model', '')].isin(model_order)]
    
    # 원본 모델명 컬럼 제거
    final_df = final_df.drop(('original_model', ''), axis=1)
    
    # 인덱스 리셋
    final_df = final_df.reset_index(drop=True)

    # 각 metric 열에 대해 최대값 강조
    metric_cols = [col for col in final_df.columns if col[1] in ['P','R','F1']]

    # 숫자 포맷과 스타일 적용
    format_dict = {}
    for col in final_df.columns:
        if col[1] in ['P', 'R', 'F1'] or col[0] == 'Acc%':
            format_dict[col] = f'{{:.{round_digits}f}}'

    styled_df = (final_df.style
                .format(format_dict)
                .apply(highlight_max, style='bold', subset=metric_cols))

    # 최종 테이블을 이미지 파일로 저장
    dfi.export(styled_df, 
             output_name + ".png",
             dpi=600  # 해상도 증가
    )
