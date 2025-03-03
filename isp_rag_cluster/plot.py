import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_grouped_bar_chart(csv_path, save_path="model_comparison.png"):
    """
    CSV 파일을 읽고, model name별로 precision, recall, f1-score에 대한 
    grouped bar chart를 subplot으로 생성 및 저장하는 함수.

    Parameters:
    - csv_path: str, CSV 파일 경로
    - save_path: str, 저장할 이미지 파일 경로 (기본값: "model_comparison.png")
    """
    # Seaborn 스타일 설정
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    
    # 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    df['model name'] = df['model name'].fillna(method='ffill')
    df = df.dropna(subset=['template type'])

    # 모델별 그룹화
    models = df['model name'].unique()
    metrics = ['precision', 'recall', 'f1-score']
    template_types = df['template type'].unique()
    
    num_models = len(models)
    num_metrics = len(metrics)
    
    # ML 연구에서 자주 사용되는 색상 팔레트
    colors = sns.color_palette("Set2", n_colors=len(template_types))
    
    # Subplot 생성
    fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 4.5, 5), sharey=True)
    fig.suptitle('Model Performance Comparison', fontsize=14, y=1.05)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(num_models)
        width = 0.2
        offsets = np.linspace(-width, width, len(template_types))
        
        for j, temp_type in enumerate(template_types):
            subset = df[df['template type'] == temp_type]
            values = [subset[subset['model name'] == model][metric].values[0] 
                     if model in subset['model name'].values else 0 for model in models]
            
            # 바 차트 그리기
            bars = ax.bar(x + offsets[j], values, width, label=temp_type, 
                         color=colors[j], alpha=0.8, edgecolor='black', linewidth=1)
            
            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)

        # 축 설정
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title(metric.replace('-', ' ').title())
        ax.set_xlabel("Model")
        if i == 0:  # y 레이블은 첫 번째 subplot에만
            ax.set_ylabel("Score")
        
        # 그리드 설정
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)  # 그리드를 바 뒤로
        
        # y축 범위 설정 (0-1)
        ax.set_ylim(0, 1.1)
        
        # 범례 설정
        if i == num_metrics - 1:  # 마지막 subplot에만 범례 표시
            ax.legend(title="Template Type", bbox_to_anchor=(1.05, 1), 
                     loc='upper left', borderaxespad=0.)

    # 레이아웃 조정
    plt.tight_layout()
    
    # 높은 DPI로 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 사용 예시
plot_grouped_bar_chart("result.csv", "grouped_bar_chart.png")