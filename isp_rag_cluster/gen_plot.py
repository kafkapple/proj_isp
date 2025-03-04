import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging

# 전역 logger 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def plot_grouped_bar_chart(csv_path, save_path="model_comparison.png"):
    """
    CSV 파일을 읽고, 각 모델별로 프롬프트 타입과 메트릭(precision, recall, f1-score)을 
    비교하는 2x2 subplot 형태의 grouped bar chart를 생성 및 저장하는 함수.

    Parameters:
    - csv_path: str, CSV 파일 경로
    - save_path: str, 저장할 이미지 파일 경로 (기본값: "model_comparison.png")
    """
    # Seaborn 스타일 설정
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    
    # 폰트 설정
    TITLE_SIZE = 14  # 타이틀 폰트 크기
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = TITLE_SIZE  # 기본 타이틀 크기 설정
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    df['model name'] = df['model name'].ffill()
    df = df.dropna(subset=['template type'])

    # 데이터 그룹화
    models = df['model name'].unique()
    metrics = ['precision', 'recall', 'f1-score']
    template_types = df['template type'].unique()
    
    # 2x2 서브플롯 레이아웃 설정
    n_rows, n_cols = 2, 2
    
    # 그래프 생성 (2x2 고정 크기)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()  # 2D 배열을 1D로 변환
    
    # 메트릭별 색상
    colors = sns.color_palette("Set2", n_colors=len(metrics))
    
    # 각 모델별 서브플롯
    for idx, model in enumerate(models):
        if idx >= 4:  # 최대 4개 모델까지만 표시
            break
            
        ax = axes[idx]
        
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
    for idx in range(len(models), 4):
        axes[idx].remove()
    
    # 레이아웃 조절
    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
    
    # 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

# 사용 예시
plot_grouped_bar_chart("result2.csv", "grouped_bar_chart3.png")