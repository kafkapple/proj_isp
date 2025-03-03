import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_csv_and_generate_outputs(csv_path, save_dir="./"):
    """
    CSV 파일을 읽어 테이블을 LaTeX 및 Markdown으로 저장하고, 
    모델별 성능 그래프를 생성하여 저장하는 함수.

    Parameters:
    - csv_path: str, CSV 파일 경로
    - save_dir: str, 저장할 디렉토리 (기본값: 현재 폴더)
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
    
    # 파일 저장 경로 설정
    tex_file = f"{save_dir}/table.tex"
    md_file = f"{save_dir}/table.md"
    plot_file = f"{save_dir}/grouped_bar_chart.png"

    # 📌 1. 테이블 저장 (LaTeX, Markdown)
    df_latex = df.to_latex(index=False, caption="Model Performance Comparison", label="tab:model_performance")
    df_md = df.to_markdown(index=False)

    with open(tex_file, "w") as f:
        f.write(df_latex)

    with open(md_file, "w") as f:
        f.write(df_md)

    print(f"✅ Table saved: {tex_file}, {md_file}")

    # 📌 2. Grouped Bar Chart 생성 및 저장
    colors = ['blue', 'orange', 'green']

    # Subplot 생성
    fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 5, 5), sharey=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(num_models)  # 모델별 x 위치
        width = 0.2  # 바 너비
        offsets = np.linspace(-width, width, len(template_types))  # 그룹별 오프셋
        
        for j, temp_type in enumerate(template_types):
            subset = df[df['template type'] == temp_type]
            values = [subset[subset['model name'] == model][metric].values[0] if model in subset['model name'].values else 0 for model in models]
            ax.bar(x + offsets[j], values, width, label=temp_type, color=colors[j])

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Model Name")
        ax.set_ylabel("Score")
        ax.legend(title="Template Type")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.show()
    
    print(f"✅ Plot saved: {plot_file}")

# 사용 예시
process_csv_and_generate_outputs("result.csv", save_dir="outputs")