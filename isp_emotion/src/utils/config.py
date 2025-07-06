from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import logging

def filter_config(cfg: DictConfig) -> dict:
    """Extract specific parts of Hydra config."""
    selected_keys = ["train", "model", "dataset"]
    return {key: OmegaConf.to_container(cfg[key], resolve=True) 
            for key in selected_keys if key in cfg}

def setup_output_dir(cfg: DictConfig) -> Path:
    """Setup output directory structure"""
    output_dir = Path(cfg.dirs.outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for subdir in cfg.dirs.subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created output directory: {output_dir}")
    return output_dir 