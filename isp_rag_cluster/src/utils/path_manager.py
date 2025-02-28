from pathlib import Path
from .lmstudio_utils import get_lmstudio_model_info
import json
from datetime import datetime

def get_vector_store_path(cfg) -> Path:
    """Get vector store path with detailed information"""
    vector_store_path = Path("vector_store")
    
    # Get embedding model name based on provider
    if cfg.model.provider == "openai":
        model_name = cfg.model.openai.embedding_model
    else:  # lmstudio
        model_info = get_lmstudio_model_info(
            base_url=cfg.model.lmstudio.base_url,
            api_key=cfg.model.lmstudio.api_key,
            model_type="embedding"
        )
        model_name = model_info.get("id", cfg.model.lmstudio.embedding_model)
    
    # Create path
    path = vector_store_path / "_".join([
        cfg.data.name,
        "all" if cfg.data.n_samples == -1 else f"n{cfg.data.n_samples}",
        cfg.model.provider,
        model_name.replace('/', '_')
    ])
    
    # Save metadata (single source of truth)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset": cfg.data.name,
            "samples": "all" if cfg.data.n_samples == -1 else f"n{cfg.data.n_samples}",
            "provider": cfg.model.provider,
            "embedding_model": model_name,
            "created_at": datetime.now().isoformat()
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    return path 