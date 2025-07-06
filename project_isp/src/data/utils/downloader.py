from pathlib import Path
import requests
import zipfile
import logging
from omegaconf import DictConfig
from tqdm import tqdm

class DataDownloader:
    """데이터셋 다운로드 클래스"""
    
    DATASET_URLS = {
        "ravdess": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
        # 다른 데이터셋 URL들...
    }
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.root_dir = Path(config.dataset.root_dir)
    
    def download_dataset(self, dataset_name: str) -> bool:
        """데이터셋 다운로드"""
        if dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # 이미 다운로드된 경우 체크
        if self._check_dataset_exists(dataset_name):
            logging.info(f"{dataset_name} dataset already exists")
            return True
            
        return self._download_and_extract(dataset_name)
    
    def _check_dataset_exists(self, dataset_name: str) -> bool:
        """데이터셋 존재 여부 확인"""
        if dataset_name == "ravdess":
            return (self.root_dir / "Actor_01").exists()
        return False
    
    def _download_and_extract(self, dataset_name: str) -> bool:
        """데이터셋 다운로드 및 압축 해제"""
        url = self.DATASET_URLS[dataset_name]
        zip_path = self.root_dir / f"{dataset_name}.zip"
        
        # 디렉토리 생성
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드
        if not zip_path.exists():
            logging.info(f"Downloading {dataset_name} dataset...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        
        # 압축 해제
        logging.info(f"Extracting {dataset_name} dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
        
        # 압축 파일 삭제
        zip_path.unlink()
        
        return True 