import requests
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_isear_dataset(cfg, save_dir: str = "data") -> str:
    """데이터셋을 다운로드합니다.
    
    Args:
        cfg: 설정 객체
        save_dir: 기본 데이터 디렉토리
        
    Returns:
        생성된 CSV 파일의 경로
    """
    # 데이터셋별 저장 디렉토리 생성
    dataset_dir = Path(save_dir) / cfg.data.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = dataset_dir / cfg.data.csv_file
    
    # 이미 파일이 존재하면 스킵
    if csv_path.exists():
        logger.info(f"데이터셋이 이미 존재합니다: {csv_path}")
        return str(csv_path)
    
    if cfg.data.name == "isear":
        # ISEAR 데이터셋 다운로드
        url = cfg.data.datasets.isear.urls[0]
        try:
            logger.info(f"ISEAR 데이터셋 다운로드 중... (URL: {url})")
            response = requests.get(url)
            response.raise_for_status()
            
            # 파일로 직접 저장
            with open(csv_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"데이터셋이 {csv_path}에 저장되었습니다.")
            return str(csv_path)
            
        except Exception as e:
            error_msg = f"데이터 다운로드 실패: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    else:
        error_msg = f"지원하지 않는 데이터셋입니다: {cfg.data.name}"
        logger.error(error_msg)
        raise ValueError(error_msg)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/config.yaml")
    download_isear_dataset(cfg)
    

    
     


