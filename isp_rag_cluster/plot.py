import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_bar_chart(csv_path, save_path="model_comparison.png"):
    """
    CSV 파일을 읽고, model name별로 precision, recall, f1-score에 대한 
    grouped bar chart를 subplot으로 생성 및 저장하는 함수.

    Parameters:
    - csv_path: str, CSV 파일 경로
    - save_path: str, 저장할 이미지 파일 경로 (기본값: "model_comparison.png")
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    # 'model name'이 빈 값이 아닌 경우를 기준으로 그룹화
    df['model name'] = df['model name'].fillna(method='ffill')

    # 'template type' 빈 값 제거
    df = df.dropna(subset=['template type'])

    # 모델별 그룹화
    models = df['model name'].unique()
    metrics = ['precision', 'recall', 'f1-score']
    template_types = df['template type'].unique()
    
    num_models = len(models)
    num_metrics = len(metrics)
    
    # 색상 정의
    colors = ['blue', 'orange', 'green']
    
    # Subplot 생성
    fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 5, 5), sharey=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(num_models)  # 모델별 x 위치
        width = 0.2  # 바 너비
        offsets = np.linspace(-width, width, len(template_types))  # 그룹별 오프셋
        
        for j, temp_type in enumerate(template_types):
            # 특정 template type 필터링
            subset = df[df['template type'] == temp_type]
            
            # 모델별 값 가져오기
            values = [subset[subset['model name'] == model][metric].values[0] if model in subset['model name'].values else 0 for model in models]
            
            # 바 차트 그리기
            ax.bar(x + offsets[j], values, width, label=temp_type, color=colors[j])

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Model Name")
        ax.set_ylabel("Score")
        ax.legend(title="Template Type")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# 사용 예시
plot_grouped_bar_chart("result.csv", "grouped_bar_chart.png")
