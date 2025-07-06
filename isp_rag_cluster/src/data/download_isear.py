import requests
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_isear_dataset(cfg, save_dir: str = "data") -> str:
    """Download the dataset.
    
    Args:
        cfg: Configuration object
        save_dir: Default data directory
        
    Returns:
        Path to the created CSV file
    """
    # Create directory for dataset-specific files
    dataset_dir = Path(save_dir) / cfg.data.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = dataset_dir / cfg.data.csv_file
    
    # If file already exists, skip
    if csv_path.exists():
        logger.info(f"Dataset already exists: {csv_path}")
        return str(csv_path)
    
    if cfg.data.name == "isear":
        # Download ISEAR dataset
        url = cfg.data.datasets.isear.urls[0]
        try:
            logger.info(f"Downloading ISEAR dataset... (URL: {url})")
            response = requests.get(url)
            response.raise_for_status()
            
            # Save directly to file
            with open(csv_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"Dataset saved to {csv_path}")
            return str(csv_path)
            
        except Exception as e:
            error_msg = f"Failed to download dataset: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    else:
        error_msg = f"Unsupported dataset: {cfg.data.name}"
        logger.error(error_msg)
        raise ValueError(error_msg)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/config.yaml")
    download_isear_dataset(cfg)
    

    
     


