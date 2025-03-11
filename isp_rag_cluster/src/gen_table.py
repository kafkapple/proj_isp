import os
import re
import pandas as pd
import numpy as np
import dataframe_image as dfi


prompt_types = ['baseline_prompt', 'zero_shot_prompt', 'few_shot_prompt',   'custom_prompt', 'rag_prompt']
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
             name_mapping = None):
    """
    name_mapping: dict, optional
        모델명과 프롬프트 타입을 변경하기 위한 매핑 딕셔너리
        예: {'qwen2.5_3b': 'Qwen 2.5', 'baseline_prompt': 'Baseline'}
    """
    # 폴더 내 CSV 파일 순회
    all_data = []
    for file_name in os.listdir(csv_dir):
        if file_name.endswith(".csv"):
            # 파일명에서 "classification_report_" 제거
            base_name = file_name.replace("classification_report_", "").replace(".csv", "")
            
            # prompt_types 리스트를 활용하여 파싱
            model = base_name
            prompt = "unknown"
            for pt in prompt_types:
                if f"_{pt}" in base_name:
                    parts = base_name.split(f"_{pt}")
                    model = parts[0]
                    prompt = pt
                    break
            
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
            row_data = {('Model', ''): model, ('Prompt', ''): prompt}
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

    # 각 metric 열(Precision, Recall, F1)에 대해 최대값 강조 (Model, Prompt, Acc% 제외)
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
