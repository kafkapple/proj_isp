
from src.gen_table import gen_table
from src.gen_plot import plot_grouped_bar_chart

if __name__ == "__main__":
    gen_table(csv_dir = "./data/classification_reports", output_name = "classification_table_new", round_digits = 0)
    # 사용 예시
    plot_grouped_bar_chart("./data/grouped_bar/result2.csv", "grouped_bar_chart_new")