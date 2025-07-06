from typing import Dict, Any, List
from pathlib import Path
import torchaudio
import torch
from src.data.datasets.base import BaseDataset
from src.data.utils.download import DatasetDownloader
import logging
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class RavdessDataset(BaseDataset):
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        super().__init__(config, split)
        
        self.config = config
        self.split = split
        
        # 기본 설정 초기화
        self.root_dir = Path(config.dataset.root_dir)
        self.sample_rate = config.dataset.audio.sample_rate
        self.duration = config.dataset.audio.duration
        self.feature_type = config.dataset.audio.feature_type
        self.n_mfcc = config.dataset.audio.n_mfcc
        self.augmentation = config.dataset.augmentation
        self.class_names = config.dataset.class_names
        
        # 데이터셋 메타데이터 로드
        self.samples = self._load_samples()
        
    def _check_dataset(self) -> bool:
        """데이터셋이 올바르게 다운로드되었는지 확인"""
        if not self.root_dir.exists():
            return False
            
        # Actor 디렉토리 확인
        actor_dirs = list(self.root_dir.glob("Actor_*"))
        if not actor_dirs:
            logging.warning(f"No Actor directories found in {self.root_dir}")
            return False
            
        # 오디오 파일 확인
        audio_files = list(self.root_dir.glob("Actor_*/*.wav"))
        if not audio_files:
            logging.warning(f"No .wav files found in {self.root_dir}")
            return False
            
        return True
        
    def _apply_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Config 설정에 따라 데이터 분할 적용"""
        split_method = self.config.dataset.get('split_method', 'random')  # 기본값은 random
        
        if split_method == 'random':
            return self._apply_random_split(df)
        elif split_method == 'kfold':
            return self._apply_kfold_split(df)
        else:
            raise ValueError(f"Unknown split method: {split_method}")

    def _apply_random_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """랜덤 분할 적용"""
        # 데이터셋 정보 출력
        logging.info("\nSplit Information:")
        logging.info(f"Total samples before split: {len(df)}")
        
        # Actor 기준 분할
        actors = np.array(sorted(df['actor'].unique()))
        n_actors = len(actors)
        logging.info(f"Total number of actors: {n_actors}")
        
        # 각 actor의 샘플 수 확인
        actor_counts = df.groupby('actor').size()
        logging.info("\nSamples per actor:")
        logging.info(f"\n{actor_counts}")
        
        np.random.seed(self.config.dataset.seed)
        np.random.shuffle(actors)
        
        train_ratio = float(self.config.dataset.splits.ratios.train)
        val_ratio = float(self.config.dataset.splits.ratios.val)
        
        # 최소 1개 이상의 actor 보장
        train_idx = max(1, int(train_ratio * n_actors))
        val_idx = max(train_idx + 1, int((train_ratio + val_ratio) * n_actors))
        
        # 인덱스 범위 검증
        if val_idx >= n_actors:
            val_idx = n_actors - 1
        
        train_actors = actors[:train_idx]
        val_actors = actors[train_idx:val_idx]
        test_actors = actors[val_idx:]
        
        # 임시 데이터프레임에 split 정보 추가
        df_split = df.copy()
        df_split['split'] = self.config.dataset.splits.train
        df_split.loc[df_split['actor'].isin(val_actors), 'split'] = self.config.dataset.splits.val
        df_split.loc[df_split['actor'].isin(test_actors), 'split'] = self.config.dataset.splits.test
        
        # 분할 결과 검증 및 로깅
        split_info = df_split.groupby('split').agg({
            'actor': lambda x: sorted(x.unique()),
            'emotion': lambda x: dict(x.value_counts())
        }).reset_index()
        
        logging.info("\nSplit details:")
        for _, row in split_info.iterrows():
            split = row['split']
            actors = row['actor']
            emotions = row['emotion']
            n_samples = sum(emotions.values())
            
            logging.info(f"\n{split}:")
            logging.info(f"  Actors ({len(actors)}): {actors}")
            logging.info(f"  Total samples: {n_samples}")
            logging.info("  Emotion distribution:")
            for emotion, count in emotions.items():
                logging.info(f"    {emotion}: {count}")
            
        # 각 분할의 샘플 수 확인
        split_counts = df_split.groupby('split').size()
        total_samples = len(df_split)
        
        logging.info("\nFinal split distribution:")
        for split_name, count in split_counts.items():
            percentage = (count/total_samples) * 100
            logging.info(f"  {split_name}: {count} samples ({percentage:.1f}%)")
        
        # 분할 결과 검증
        if 0 in split_counts.values:
            logging.warning("Warning: Some splits have no samples!")
            available_splits = df_split['split'].unique()
            logging.warning(f"Available splits: {available_splits}")
        
        return df_split

    def _apply_kfold_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """K-Fold 분할 적용"""
        k = self.config.dataset.splits.get('k_folds', 5)
        fold = self.config.dataset.splits.get('current_fold', 0)
        
        # Actor 기준 K-Fold 분할
        actors = np.array(sorted(df['actor'].unique()))
        kf = KFold(n_splits=k, shuffle=True, random_state=self.config.dataset.seed)
        
        # 현재 fold에 해당하는 분할 찾기
        for i, (train_idx, test_idx) in enumerate(kf.split(actors)):
            if i == fold:
                test_actors = actors[test_idx]
                
                # validation을 위해 train 데이터 추가 분할
                train_actors = actors[train_idx]
                val_size = int(len(train_actors) * self.config.dataset.splits.ratios.val)
                val_actors = train_actors[:val_size]
                train_actors = train_actors[val_size:]
                
                # 임시 데이터프레임에 split 정보 추가
                df_split = df.copy()
                df_split['split'] = self.config.dataset.splits.train
                df_split.loc[df_split['actor'].isin(val_actors), 'split'] = self.config.dataset.splits.val
                df_split.loc[df_split['actor'].isin(test_actors), 'split'] = self.config.dataset.splits.test
                
                return df_split
                
        raise ValueError(f"Invalid fold number: {fold}")

    def _load_samples(self) -> pd.DataFrame:
        """데이터셋 샘플 로드 및 분할"""
        metadata_path = self.root_dir / "ravdess_metadata.csv"
        
        # 기본 메타데이터 로드/생성
        if not metadata_path.exists():
            success = DatasetDownloader._generate_ravdess_metadata(self.root_dir)
            if not success:
                raise RuntimeError("Failed to generate metadata")
        
        # 메타데이터 로드
        df = pd.read_csv(metadata_path)
        
        # 감정 필터링
        df = df[df['emotion'].isin(self.class_names)]
        
        # Split 적용
        df_split = self._apply_split(df)
        
        # split 매핑 확인
        split_map = {
            'train': self.config.dataset.splits.train,
            'val': self.config.dataset.splits.val,
            'test': self.config.dataset.splits.test
        }
        
        if self.split not in split_map:
            raise ValueError(f"Invalid split: {self.split}. Must be one of {list(split_map.keys())}")
        
        target_split = split_map[self.split]
        samples = df_split[df_split['split'] == target_split]
        
        # 디버깅을 위한 로깅 추가
        logging.info(f"Total samples: {len(df)}")
        logging.info(f"Split distribution:")
        for split_name, count in df_split['split'].value_counts().items():
            logging.info(f"  {split_name}: {count} samples")
        logging.info(f"Selected split '{self.split}' (mapped to '{target_split}')")
        logging.info(f"Found {len(samples)} samples for {self.split} split")
        
        if len(samples) == 0:
            available_splits = df_split['split'].unique()
            raise ValueError(
                f"No samples found for {self.split} split (mapped to '{target_split}'). "
                f"Available splits: {available_splits}"
            )
        
        return samples.copy()
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """데이터셋에서 하나의 샘플을 가져옴"""
        sample = self.samples.iloc[index]
        
        # 오디오 로드 및 전처리
        waveform, sr = torchaudio.load(sample['file_path'])
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"1. Initial waveform shape: {waveform.shape}")
        
        # 샘플링 레이트 변환
        if sr != self.config.dataset.audio.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.dataset.audio.sample_rate)
            waveform = resampler(waveform)
            if self.config.debug.enabled and self.config.debug.log_shapes:
                logging.info(f"2. After resampling shape: {waveform.shape}")
        
        # 모노로 변환 (필요한 경우)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            if self.config.debug.enabled and self.config.debug.log_shapes:
                logging.info(f"3. After mono conversion shape: {waveform.shape}")
        
        # 길이 조정
        target_length = int(self.config.dataset.audio.duration * self.config.dataset.audio.sample_rate)
        current_length = waveform.size(1)
        
        if current_length > target_length:
            start = torch.randint(0, current_length - target_length + 1, (1,))
            waveform = waveform[:, start:start + target_length]
        elif current_length < target_length:
            pad_length = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"4. After length adjustment shape: {waveform.shape}")
        
        # Augmentation 적용
        if self.split == 'train' and self.config.dataset.augmentation.enabled:
            waveform = self._apply_augmentation(waveform)
            if self.config.debug.enabled and self.config.debug.log_shapes:
                logging.info(f"5. After augmentation shape: {waveform.shape}")
        
        # 차원 확인 및 조정
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"6. Final waveform shape: {waveform.shape}")
        
        # 레이블 준비
        label = self.class_names.index(sample['emotion'])
        
        return {
            "audio": waveform,
            "label": torch.tensor(label, dtype=torch.long)
        }

    def _generate_ravdess_metadata(root_dir: Path) -> bool:
        """RAVDESS 데이터셋의 메타데이터 생성"""
        metadata_path = root_dir / "ravdess_metadata.csv"
        
        if metadata_path.exists():
            return True
        
        # 오디오 파일 존재 여부 확인
        audio_files = list(root_dir.glob("**/*.wav"))
        if not audio_files:
            logging.error(f"No audio files found in {root_dir}")
            return False
        
        logging.info("Generating RAVDESS metadata...")
        metadata = []
        
        # 감정 매핑 (RAVDESS 코드 -> 감정 이름)
        emotion_map = {
            "01": "neutral",
            "02": "calm",      # config에 따라 포함/제외 가능
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
            
            # 파일명 파싱 (Actor_01-01-01-01-01-01-01.wav 형식)
            filename = audio_path.stem
            parts = filename.split("-")
            
            # 감정 코드 확인
            emotion_code = parts[2]
            if emotion_code not in emotion_map:
                continue
            
            metadata.append({
                'file_path': str(audio_path),
                'actor': int(parts[0].replace("Actor_", "")),
                'vocal_channel': int(parts[1]),
                'emotion': emotion_map[emotion_code],  # 숫자 대신 감정 이름 저장
                'emotion_intensity': int(parts[3]),
                'statement': int(parts[4]),
                'repetition': int(parts[5]),
                'gender': 'female' if int(parts[6]) == 2 else 'male'
            })
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(metadata)
        
        # Train/Val/Test 분할 (Actor 기준)
        actors = np.array(sorted(df['actor'].unique()))
        np.random.seed(42)
        np.random.shuffle(actors)
        
        n_actors = len(actors)
        train_actors = actors[:int(0.7 * n_actors)]
        val_actors = actors[int(0.7 * n_actors):int(0.85 * n_actors)]
        test_actors = actors[int(0.85 * n_actors):]
        
        df['split'] = 'train'
        df.loc[df['actor'].isin(val_actors), 'split'] = 'validation'
        df.loc[df['actor'].isin(test_actors), 'split'] = 'test'
        
        df.to_csv(metadata_path, index=False)
        logging.info(f"Metadata saved to {metadata_path}")
        return True

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """오디오 파형에 augmentation 적용"""
        if not self.config.dataset.augmentation.enabled or self.split != 'train':
            return waveform
        
        aug_config = self.config.dataset.augmentation
        
        # 가우시안 노이즈 추가
        if aug_config.noise.enabled:
            noise = torch.randn_like(waveform) * aug_config.noise.noise_level
            waveform = waveform + noise
        
        # 볼륨 변경
        if aug_config.volume.enabled:
            gain = torch.FloatTensor(1).uniform_(
                aug_config.volume.min_gain,
                aug_config.volume.max_gain
            )
            waveform = waveform * gain
        
        return waveform

