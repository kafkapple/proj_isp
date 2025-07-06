from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
from typing import Dict
import logging
import numpy as np

class DataProcessor:
    """데이터셋 처리를 위한 유틸리티 클래스"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.root_dir = Path(config.dataset.root_dir)
        self.filter = DataFilter()
        self.stats = DataStats()
        
        # emotion class 매핑 초기화
        self.emotion_map = self._init_emotion_map()
    
    def _init_emotion_map(self) -> Dict[str, str]:
        """Emotion 클래스 매핑 초기화"""
        if not hasattr(self.config.dataset, 'class_names'):
            raise ValueError("Dataset config must have 'class_names' defined")
            
        # RAVDESS 데이터셋의 경우 01~08 매핑
        class_names = self.config.dataset.class_names
        return {f"{i+1:02d}": name for i, name in enumerate(class_names)}
    
    def load_metadata(self, root_dir: Path) -> pd.DataFrame:
        """메타데이터 로드 또는 생성"""
        metadata_path = root_dir / "ravdess_metadata.csv"
        
        if not metadata_path.exists():
            return self._generate_metadata(root_dir)
            
        df = pd.read_csv(metadata_path)
        return df
    
    def _generate_metadata(self, root_dir: Path) -> pd.DataFrame:
        """메타데이터 생성"""
        metadata_path = root_dir / "ravdess_metadata.csv"
        
        # 오디오 파일 찾기
        audio_files = list(root_dir.rglob("*.wav"))
        
        metadata = []
        for audio_path in audio_files:
            try:
                # 파일명 파싱 (예: Actor_01/03-01-04-01-02-01-01.wav)
                filename = audio_path.stem
                parts = filename.split("-")
                actor_id = int(audio_path.parent.name.replace("Actor_", ""))
                
                emotion_code = parts[2]
                if emotion_code not in self.emotion_map:
                    logging.warning(f"Unknown emotion code {emotion_code} in {audio_path}")
                    continue
                
                metadata.append({
                    'file_path': str(audio_path.relative_to(root_dir)),
                    'actor': actor_id,
                    'vocal_channel': int(parts[1]),
                    'emotion': self.emotion_map[parts[2]],
                    'emotion_intensity': int(parts[3]),
                    'statement': int(parts[4]),
                    'repetition': int(parts[5]),
                    'gender': 'female' if actor_id % 2 == 0 else 'male'
                })
                
            except Exception as e:
                logging.warning(f"Error processing file {audio_path}: {e}")
                continue
        
        # 메타데이터를 DataFrame으로 변환
        df = pd.DataFrame(metadata)
        
        # 레이블 인코딩
        class_to_idx = {name: idx for idx, name in enumerate(self.config.dataset.class_names)}
        df['label'] = df['emotion'].map(class_to_idx)
        
        # split 컬럼 초기화
        df['split'] = ''
        
        # 저장
        df.to_csv(metadata_path, index=False)
        
        # 통계 출력
        self._log_statistics(df)
        
        return df
    
    def process_dataset(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """데이터셋 전처리 파이프라인"""
        # 필터 적용 (설정이 있는 경우에만)
        if hasattr(self.config.dataset, 'filtering') and self.config.dataset.filtering.enabled:
            df = self.filter.apply_filters(df, self.config.dataset.filtering)
            
        # split 적용 (split 컬럼이 비어있거나 없는 경우)
        if 'split' not in df.columns or df['split'].isna().all():
            df = self._apply_split(df)
            # 변경된 split 정보 저장
            metadata_path = Path(self.config.dataset.root_dir) / "ravdess_metadata.csv"
            df.to_csv(metadata_path, index=False)
            logging.info(f"Updated split information saved to {metadata_path}")
        
        # 현재 split에 해당하는 데이터만 필터링
        split_df = df[df['split'] == split].copy()
        
        if len(split_df) == 0:
            # split 정보 로깅
            logging.error(f"\nSplit distribution:")
            logging.error(df['split'].value_counts())
            raise ValueError(f"No samples found for {split} split!")
            
        # 클래스 밸런싱 (train split만)
        if split == 'train' and hasattr(self.config.dataset, 'balance') and self.config.dataset.balance.enabled:
            split_df = self._balance_classes(split_df)
            
        # 통계 생성 및 로깅
        logging.info(f"\n{split} split statistics:")
        self._log_statistics(split_df)
        
        return split_df
    
    def _apply_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터셋 분할 적용"""
        # Actor 기준으로 분할
        actors = np.array(sorted(df['actor'].unique()))
        np.random.seed(self.config.dataset.seed)
        np.random.shuffle(actors)
        
        n_actors = len(actors)
        train_ratio = 0.7  # 기본값
        val_ratio = 0.15   # 기본값
        
        # split 비율이 설정에 있으면 사용
        if hasattr(self.config.dataset, 'splits'):
            train_ratio = self.config.dataset.splits.ratios.train
            val_ratio = self.config.dataset.splits.ratios.val
        
        n_train = int(n_actors * train_ratio)
        n_val = int(n_actors * val_ratio)
        
        train_actors = actors[:n_train]
        val_actors = actors[n_train:n_train + n_val]
        test_actors = actors[n_train + n_val:]
        
        # split 정보 설정
        df.loc[df['actor'].isin(train_actors), 'split'] = 'train'
        df.loc[df['actor'].isin(val_actors), 'split'] = 'val'
        df.loc[df['actor'].isin(test_actors), 'split'] = 'test'
        
        # split 분포 로깅
        logging.info("\nDataset split distribution:")
        for split_name, count in df['split'].value_counts().items():
            logging.info(f"{split_name}: {count} samples ({count/len(df)*100:.2f}%)")
        
        return df
    
    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """클래스 밸런싱"""
        if not hasattr(self.config.dataset.balance, 'method'):
            return df
            
        method = self.config.dataset.balance.method
        if method == 'oversample':
            # 오버샘플링 구현
            pass
        elif method == 'undersample':
            # 언더샘플링 구현
            pass
            
        return df
    
    def _log_statistics(self, df: pd.DataFrame) -> None:
        """데이터셋 통계 출력"""
        logging.info(f"\nDataset Statistics:")
        logging.info(f"Total samples: {len(df)}")
        logging.info("\nEmotion distribution:")
        logging.info(df['emotion'].value_counts())
        logging.info("\nGender distribution:")
        logging.info(df['gender'].value_counts())
        logging.info("\nActor distribution:")
        logging.info(df['actor'].value_counts())

class DataFilter:
    """데이터 필터링 클래스"""
    
    def apply_filters(self, df: pd.DataFrame, 
                     filter_config: DictConfig) -> pd.DataFrame:
        """필터 적용"""
        if not filter_config.enabled:
            return df
            
        df = self._filter_by_speech(df, filter_config)
        df = self._filter_by_emotion(df, filter_config)
        df = self._filter_by_gender(df, filter_config)
        return df
    
    def _filter_by_speech(self, df: pd.DataFrame, 
                         config: DictConfig) -> pd.DataFrame:
        """음성/노래 필터링"""
        if hasattr(config, 'speech_only') and config.speech_only:
            return df[df['vocal_channel'] == 1]
        return df
    
    def _filter_by_emotion(self, df: pd.DataFrame, 
                          config: DictConfig) -> pd.DataFrame:
        """감정 필터링"""
        if hasattr(config, 'emotions'):
            if config.emotions.include:
                df = df[df['emotion'].isin(config.emotions.include)]
            if config.emotions.exclude:
                df = df[~df['emotion'].isin(config.emotions.exclude)]
        return df
    
    def _filter_by_gender(self, df: pd.DataFrame, 
                         config: DictConfig) -> pd.DataFrame:
        """성별 필터링"""
        if hasattr(config, 'gender') and config.gender:
            return df[df['gender'].isin(config.gender)]
        return df

class DataStats:
    """데이터 통계 분석 클래스"""
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """통계 생성"""
        stats = {
            "total_samples": len(df),
            "class_distribution": self._get_class_distribution(df),
            "gender_distribution": self._get_gender_distribution(df),
            "actor_distribution": self._get_actor_distribution(df),
            "emotion_intensity_distribution": self._get_intensity_distribution(df)
        }
        return stats
    
    def _get_class_distribution(self, df: pd.DataFrame) -> Dict:
        """클래스 분포"""
        return df['emotion'].value_counts().to_dict()
    
    def _get_gender_distribution(self, df: pd.DataFrame) -> Dict:
        """성별 분포"""
        return df['gender'].value_counts().to_dict()
    
    def _get_actor_distribution(self, df: pd.DataFrame) -> Dict:
        """배우별 분포"""
        return df['actor'].value_counts().to_dict()
    
    def _get_intensity_distribution(self, df: pd.DataFrame) -> Dict:
        """감정 강도 분포"""
        return df['emotion_intensity'].value_counts().to_dict() 