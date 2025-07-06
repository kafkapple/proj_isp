import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging

# 전역 logger 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def plot_grouped_bar_chart(csv_path, save_path="model_comparison", name_mapping=None, model_order=None, prompt_order=None):
    """
    CSV 파일을 읽고, 각 모델별로 프롬프트 타입과 메트릭을 비교하는 grouped bar chart를 생성.

    Parameters:
    - csv_path: str, CSV 파일 경로
    - save_path: str, 저장할 이미지 파일 경로
    - name_mapping: dict, optional
        모델명과 프롬프트 타입을 변경하기 위한 매핑 딕셔너리
        예: {'qwen2.5_3b': 'Qwen 2.5', 'baseline_prompt': 'Baseline'}
    - model_order: list, optional
        모델 순서를 지정하는 리스트. 매핑 이전의 원래 모델명 사용
        예: ['qwen2.5_3b', 'llama3.2', 'mistral']
    """
    # Seaborn 스타일 설정
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    
    # 폰트 설정
    TITLE_SIZE = 14
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    print(f"\nLoaded DataFrame from {csv_path}")
    print("Original DataFrame:")
    print(df)
    
    # 매핑 적용 전에 모델 순서 처리를 위한 원본 컬럼 저장
    df['original_model'] = df['model name']
    df['original_template'] = df['template type']
    
    # 매핑 적용
    if name_mapping:
        print("\nApplying name mapping...")
        df['model name'] = df['model name'].map(lambda x: name_mapping.get(x, x))
        df['template type'] = df['template type'].map(lambda x: name_mapping.get(x, x))
        print("After mapping:")
        print(df[['original_model', 'model name', 'original_template', 'template type']])
    
    df['model name'] = df['model name'].ffill()
    df = df.dropna(subset=['template type'])

    # 데이터 그룹화 및 모델 순서 적용
    if model_order:
        print("\nApplying model order...")
        # 매핑된 이름으로 변환
        ordered_models = [name_mapping.get(m, m) if name_mapping else m for m in model_order]
        print(f"Ordered models: {ordered_models}")
        # 지정된 순서에 있는 모델만 선택하고 순서 유지
        df = df[df['original_model'].isin(model_order)]
        df['model name'] = pd.Categorical(df['model name'], ordered_models, ordered=True)
        df = df.sort_values('model name')
        models = ordered_models
    else:
        models = df['model name'].unique()
    
    print("\nUnique models after ordering:")
    print(models)
    
    metrics = ['precision', 'recall', 'f1-score']
    
    # 프롬프트 타입 순서 지정 (매핑 적용)
    if prompt_order:
        print("\nApplying prompt order...")
        template_types = [name_mapping.get(pt, pt) if name_mapping else pt for pt in prompt_order]
        print(f"Template types after mapping: {template_types}")
        # 데이터프레임의 프롬프트 순서도 조정
        df['template type'] = pd.Categorical(df['template type'], template_types, ordered=True)
        df = df.sort_values(['model name', 'template type'])
    else:
        template_types = df['template type'].unique()
    
    print("\nFinal DataFrame after all ordering:")
    print(df[['model name', 'template type', 'precision', 'recall', 'f1-score']])
    
    # subplot 레이아웃 계산
    n_models = len(models)
    n_cols = min(3, n_models)  # 최대 3열, 모델이 더 적으면 모델 수만큼
    n_rows = (n_models + n_cols - 1) // n_cols  # 올림 나눗셈으로 필요한 행 수 계산
    
    # 그래프 생성
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])  # 단일 subplot인 경우
    elif n_rows == 1:
        axes = np.array([axes])  # 1행인 경우
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])  # 1열인 경우
    
    # 메트릭별 색상
    colors = sns.color_palette("Set2", n_colors=len(metrics))
    
    # 각 모델별 서브플롯
    for idx, model in enumerate(models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        x = np.arange(len(template_types))
        width = 0.25  # bar width
        
        model_data = df[df['model name'] == model]
        
        for i, metric in enumerate(metrics):
            values = []
            for temp_type in template_types:
                subset = model_data[model_data['template type'] == temp_type]
                value = subset[metric].values[0] if not subset.empty else 0
                values.append(value)
            
            # 바 차트 그리기
            bars = ax.bar(x + (i-1)*width, values, width, 
                         label=metric.replace('-', ' ').title(),
                         color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
            
            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        # 축 설정
        ax.set_xticks(x)
        ax.set_xticklabels(template_types, rotation=45, ha='right')
        ax.set_ylabel("Score")
        ax.set_title(model, pad=10, fontsize=TITLE_SIZE, fontweight='bold')  # 타이틀 폰트 크기와 굵기 설정
        
        # 그리드 설정
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # y축 범위 설정
        ax.set_ylim(0, 1.1)
        
        # 상단 경계선 제거
        ax.spines['top'].set_visible(False)
        
        # 첫 번째 subplot에만 범례 표시
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    # 남은 서브플롯 제거
    for row in range(n_rows):
        for col in range(n_cols):
            if row * n_cols + col >= len(models):
                axes[row, col].remove()
    
    # 레이아웃 조절
    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
    
    # 저장
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

