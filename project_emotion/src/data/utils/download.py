# src/data/utils/download.py
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import kaggle
from dotenv import load_dotenv
import requests
import shutil

class DatasetDownloader:
    DATASET_URLS = {
        "ravdess": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
        "fer2013": "fer2013/fer2013/fer2013.csv"  # Kaggle dataset path
    }
    
    _instance = None
    _downloaded = set()  # 다운로드 완료된 데이터셋 추적
    
    @classmethod
    def download_and_extract(cls, dataset_name: str, root_dir: str):
        root_dir = Path(root_dir)
        
        # 더 엄격한 존재 여부 체크
        if dataset_name == "ravdess":
            if (root_dir / "Actor_01").exists() and any((root_dir / "Actor_01").iterdir()):
                logging.info(f"RAVDESS dataset already exists at {root_dir}")
                return
        elif dataset_name == "fer2013":
            if (root_dir / "train").exists() and any((root_dir / "train").iterdir()):
                logging.info(f"FER2013 dataset already exists at {root_dir}")
                return
                
        # 데이터셋별 다운로드 로직
        if dataset_name == "ravdess":
            cls._download_ravdess(root_dir)
        elif dataset_name == "fer2013":
            cls._download_fer2013(root_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup Kaggle credentials
        self._setup_kaggle_credentials()
        
    def _setup_kaggle_credentials(self):
        """Setup Kaggle credentials from environment variables"""
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        
        if not (kaggle_username and kaggle_key):
            logging.warning(
                "Kaggle credentials not found in .env file. "
                "Please set KAGGLE_USERNAME and KAGGLE_KEY"
            )
            return
            
        # Create kaggle.json if it doesn't exist
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        kaggle_cred = kaggle_dir / 'kaggle.json'
        if not kaggle_cred.exists():
            kaggle_cred.write_text(
                f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}'
            )
            # Restrict access to prevent warnings
            kaggle_cred.chmod(0o600)
            
    @staticmethod
    def _download_fer2013(root_dir: Path):
        csv_path = root_dir / "fer2013.csv"
        
        if not csv_path.exists():
            logging.info("Downloading FER2013 dataset...")
            try:
                # Download using Kaggle API
                kaggle.api.authenticate()
                kaggle.api.dataset_download_file(
                    dataset="deadskull7/fer2013",
                    file_name="fer2013.csv",
                    path=root_dir
                )
            except Exception as e:
                logging.error(f"Error downloading FER2013: {e}")
                logging.info(
                    "Please ensure you have set KAGGLE_USERNAME and KAGGLE_KEY "
                    "in your .env file or manually download from: "
                    "https://www.kaggle.com/datasets/deadskull7/fer2013"
                )
                raise
                
        # Convert CSV to more efficient format
        if not (root_dir / "processed").exists():
            logging.info("Processing FER2013 dataset...")
            DatasetDownloader._process_fer2013(csv_path, root_dir)
            
    @staticmethod
    def _process_fer2013(csv_path: Path, root_dir: Path):
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Create directories for each split
        processed_dir = root_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        for split in ["Training", "PublicTest", "PrivateTest"]:
            split_dir = processed_dir / split.lower()
            split_dir.mkdir(exist_ok=True)
            
            # Filter data for current split
            split_data = df[df['Usage'] == split]
            
            # Process each image and save
            for idx, row in split_data.iterrows():
                emotion = row['emotion']
                pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                img = pixels.reshape(48, 48)
                
                # Save as numpy array (more efficient than image files for this dataset)
                emotion_dir = split_dir / str(emotion)
                emotion_dir.mkdir(exist_ok=True)
                np.save(emotion_dir / f"{idx}.npy", img)

    @staticmethod
    def _download_ravdess(root_dir: Path):
        try:
            # Kaggle API 인증
            load_dotenv()
            os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
            
            if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
                raise ValueError("Kaggle credentials not found in .env file")
            
            # 데이터셋 다운로드
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'uwrfkaggler/ravdess-emotional-speech-audio',
                path=str(root_dir),
                unzip=True
            )
            
            # audio_speech_actors_01-24 폴더에서 파일들을 root_dir로 이동
            src_dir = root_dir / "audio_speech_actors_01-24"
            if src_dir.exists():
                for item in src_dir.iterdir():
                    if item.is_dir():
                        shutil.move(str(item), str(root_dir))
                shutil.rmtree(str(src_dir))
            
            logging.info(f"Dataset downloaded and extracted to {root_dir}")
            
        except Exception as e:
            logging.error(f"Error downloading RAVDESS dataset: {str(e)}")
            raise

    @staticmethod
    def _generate_ravdess_metadata(root_dir: Path) -> bool:
        """RAVDESS 데이터셋의 기본 메타데이터 생성"""
        metadata_path = root_dir / "ravdess_metadata.csv"
        
        # 오디오 파일 존재 여부 확인
        audio_files = list(root_dir.glob("Actor_*//*.wav"))  # Actor 폴더 경로 수정
        if not audio_files:
            logging.error(f"No audio files found in {root_dir}")
            return False
            
        logging.info(f"Found {len(audio_files)} audio files")
        
        # 데이터셋 기본 정보 수집
        actors = set()
        emotions = set()
        metadata = []
        
        # 감정 매핑
        emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        
        for audio_path in audio_files:
            if "Actor_" not in str(audio_path):
                continue
                
            try:
                # 파일명 파싱 (Actor_01/03-01-04-01-02-01-01.wav 형식)
                actor_id = int(audio_path.parent.name.replace("Actor_", ""))
                filename = audio_path.stem
                parts = filename.split("-")
                
                emotion_code = parts[2]
                if emotion_code not in emotion_map:
                    continue
                    
                emotion = emotion_map[emotion_code]
                
                actors.add(actor_id)
                emotions.add(emotion)
                
                metadata.append({
                    'file_path': str(audio_path),
                    'actor': actor_id,
                    'vocal_channel': int(parts[1]),
                    'emotion': emotion,
                    'emotion_intensity': int(parts[3]),
                    'statement': int(parts[4]),
                    'repetition': int(parts[5]),
                    'gender': 'female' if int(parts[6]) == 2 else 'male'
                })
                
            except (IndexError, ValueError) as e:
                logging.warning(f"Error processing file {audio_path}: {e}")
                continue
        
        if not metadata:
            logging.error("No valid audio files found")
            return False
        
        # 데이터셋 기본 정보 출력
        df = pd.DataFrame(metadata)
        logging.info("\nRAVDESS Dataset Statistics:")
        logging.info(f"Total number of audio files: {len(df)}")
        logging.info(f"Number of actors: {len(actors)}")
        logging.info(f"Emotions: {sorted(emotions)}")
        
        # 배우별 감정 분포
        actor_emotion_dist = pd.crosstab(df['actor'], df['emotion'])
        logging.info("\nEmotion distribution per actor:")
        logging.info(f"\n{actor_emotion_dist}")
        
        # 성별 분포
        gender_dist = df['gender'].value_counts()
        logging.info("\nGender distribution:")
        logging.info(f"\n{gender_dist}")
        
        # 감정별 샘플 수
        emotion_dist = df['emotion'].value_counts()
        logging.info("\nEmotion distribution:")
        logging.info(f"\n{emotion_dist}")
        
        # 메타데이터 저장
        df.to_csv(metadata_path, index=False)
        logging.info(f"\nMetadata saved to {metadata_path}")
        
        return True

    def download_fer2013(self, root_dir: Path) -> bool:
        """FER2013 데이터셋 다운로드"""
        try:
            import kaggle
            
            # 데이터셋 경로 생성
            dataset_path = root_dir / "fer2013"
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Kaggle에서 데이터셋 다운로드
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'msambare/fer2013',
                path=str(dataset_path),
                unzip=True
            )
            
            # 메타데이터 생성
            return self._generate_fer2013_metadata(dataset_path)
            
        except Exception as e:
            logging.error(f"Error downloading FER2013 dataset: {e}")
            return False

    def _generate_fer2013_metadata(self, root_dir: Path) -> bool:
        """FER2013 메타데이터 생성"""
        try:
            metadata_path = root_dir / "fer2013_metadata.csv"
            
            # CSV 파일 읽기
            df = pd.read_csv(root_dir / "fer2013.csv")
            
            # 데이터셋 통계 출력
            logging.info("\nFER2013 Dataset Statistics:")
            logging.info(f"Total samples: {len(df)}")
            logging.info("\nClass distribution:")
            logging.info(df['emotion'].value_counts())
            
            # 메타데이터 저장
            df.to_csv(metadata_path, index=False)
            logging.info(f"\nMetadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error generating FER2013 metadata: {e}")
            return False

# src/data/validation.py
class DatasetValidator:
    @staticmethod
    def validate_fer2013(root_dir: Path) -> bool:
        processed_dir = root_dir / "processed"
        if not processed_dir.exists():
            return False
            
        # Check for expected splits
        expected_splits = ["training", "publictest", "privatetest"]
        for split in expected_splits:
            split_dir = processed_dir / split
            if not split_dir.exists():
                return False
                
            # Check for emotion class directories (0-6)
            for emotion in range(7):
                emotion_dir = split_dir / str(emotion)
                if not emotion_dir.exists():
                    return False
                    
                # Check if directory contains any files
                if not list(emotion_dir.glob("*.npy")):
                    return False
                    
        return True