from pathlib import Path
from .lmstudio_utils import get_lmstudio_model_info

def get_vector_store_path(cfg) -> Path:
    """Get vector store path with detailed information"""
    vector_store_path = Path("vector_store")
    
    # Get embedding model info and name
    if cfg.model.provider == "openai":
        # OpenAI의 경우 config에 설정된 모델명 사용
        model_name = cfg.model.openai.embedding_model
    elif cfg.model.provider == "lmstudio":
        # LMStudio의 경우 실제 로드된 모델 ID 사용
        model_info = get_lmstudio_model_info(
            base_url=cfg.model.lmstudio.base_url,
            api_key=cfg.model.lmstudio.api_key
        )
        model_name = model_info.get("id", "unknown")
    else:
        # 기타 provider의 경우 기본 embedding 모델명 사용
        model_name = cfg.model.embeddings.default_model
        
    # Create path with detailed information
    path_components = [
        cfg.data.name,  # Dataset name
        "all" if cfg.data.n_samples == -1 else f"n{cfg.data.n_samples}",  # "all" for full dataset
        cfg.model.provider,  # Provider
        model_name.replace('/', '_')  # Model name (cleaned)
    ]
    
    # Print debug info
    print(f"\nVector store path components: {path_components}")
    path = vector_store_path / "_".join(path_components)
    print(f"Full vector store path: {path}")
    print(f"Path exists: {path.exists()}")
    
    return path 