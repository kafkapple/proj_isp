from omegaconf import OmegaConf
from src.eval import save_metrics
import pandas as pd
from pathlib import Path

def metric():
    cfg = OmegaConf.load("config/llm.yaml")
    
    # Convert path to Path object for correct path construction
    base_path = Path("D:/dev/isp_rag_cluster/outputs")
    
    # Verify actual existence of directory
    output_folders = [f for f in base_path.iterdir() if f.is_dir()]
    if not output_folders:
        raise FileNotFoundError("No output folders found")
    
    # Select latest folder (based on timestamp)
    latest_folder = max(output_folders, key=lambda x: x.stat().st_mtime)
    latest_folder = Path(r"outputs\20250303_172613_qwen2.5_emotion_prompt\dataset-isear_model-qwen2.5")
    # File path construction
    model_name = latest_folder.name.split("_model-")[1]
    #model_name = "llama3.2"  # Or other model name
    
    file_name = f"dataset-isear_model-{model_name}.csv"
    latest_folder = latest_folder.parent
    file_path = latest_folder / file_name
    
    print(f"Looking for file at: {file_path}")  # For debugging
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read file
    df_isear = pd.read_csv(file_path)
    
    # Save metrics
    save_metrics(df_isear, cfg, model_name, latest_folder)


if __name__ == "__main__":
    metric()
