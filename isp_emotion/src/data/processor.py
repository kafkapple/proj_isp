from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from typing import Dict, Any
import logging
from src.data.metadata import MetadataGenerator
import requests
import zipfile
import os
import shutil
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from src.data.dataset import EmotionDataset
import numpy as np

class DatasetDownloader(ABC):
    @abstractmethod
    def download(self, root_dir: Path):
        pass

class RavdessDownloader(DatasetDownloader):
    def download(self, root_dir: Path):
        """RAVDESS dataset download"""
        url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        zip_path = root_dir / "ravdess.zip"
        
        # Create download directory
        root_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # download and unzip logic
            self._download_file(url, zip_path)
            self._extract_zip(zip_path, root_dir)
            self._organize_files(root_dir)
            logging.info("RAVDESS dataset download completed!")
        except Exception as e:
            logging.error(f"Error downloading RAVDESS dataset: {e}")
            raise

    def _download_file(self, url: str, zip_path: Path):
        """file download"""
        logging.info("Downloading RAVDESS dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def _extract_zip(self, zip_path: Path, root_dir: Path):
        """unzip"""
        logging.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        
        # Try to find and move files if Audio_Speech_Actors directory exists
        speech_dirs = list(root_dir.glob("Audio_Speech_Actors*"))
        if speech_dirs:
            speech_dir = speech_dirs[0]
            for actor_dir in speech_dir.glob("Actor_*"):
                if actor_dir.is_dir():
                    # Move entire actor directory to root_dir
                    target_dir = root_dir / actor_dir.name
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.move(str(actor_dir), str(target_dir))
        
            # Remove Audio_Speech_Actors directory
            shutil.rmtree(speech_dir)
        
        # Always delete zip file
        zip_path.unlink()

    def _organize_files(self, root_dir: Path):
        """Verify file structure"""
        # Check if all actor directories exist
        actor_dirs = list(root_dir.glob("Actor_*"))
        if len(actor_dirs) != 24:  # RAVDESS has 24 actors
            raise ValueError(f"Expected 24 actor directories, found {len(actor_dirs)}")
        
        # Verify each actor directory has wav files
        for actor_dir in actor_dirs:
            wav_files = list(actor_dir.glob("*.wav"))
            if not wav_files:
                raise ValueError(f"No wav files found in {actor_dir}")
        
        logging.info(f"Found {len(actor_dirs)} actor directories")

class DataProcessor:
    DOWNLOADERS = {
        "ravdess": RavdessDownloader,
        # add other datasets:
        # "iemocap": IemocapDownloader,
        # "emodb": EmoDBDownloader,
    }
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.root_dir = Path(config.dataset.root_dir)
        self.metadata_generator = MetadataGenerator(config)
        
        # if dataset not exists, download
        if not self.root_dir.exists() or not any(self.root_dir.iterdir()):
            self.download_dataset()
        
        # Generate metadata file if it doesn't exist
        metadata_path = self.root_dir / "metadata.csv"
        if not metadata_path.exists():
            df = self.metadata_generator.generate(self.root_dir)
            df.to_csv(metadata_path, index=False)
            logging.info(f"Generated metadata file: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)

    def create_dataloaders(self):
        """Create train, validation and test dataloaders"""
        # Load and split data
        train_data, val_data, test_data = self._split_data()
        
        # Create datasets with split information
        train_dataset = EmotionDataset(
            metadata=train_data,
            config=OmegaConf.merge(self.config, {"dataset": {"split": "train"}})
        )
        
        val_dataset = EmotionDataset(
            metadata=val_data,
            config=OmegaConf.merge(self.config, {"dataset": {"split": "val"}})
        )
        
        test_dataset = EmotionDataset(
            metadata=test_data,
            config=OmegaConf.merge(self.config, {"dataset": {"split": "test"}})
        )
        
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.config.train.batch_size,
                shuffle=True,
                num_workers=self.config.train.num_workers,
                pin_memory=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=self.config.train.batch_size,
                shuffle=False,
                num_workers=self.config.train.num_workers,
                pin_memory=True
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=self.config.train.batch_size,
                shuffle=False,
                num_workers=self.config.train.num_workers,
                pin_memory=True
            )
        }
        
        return dataloaders
    
    def download_dataset(self):
        """download dataset"""
        dataset_name = self.config.dataset.name.lower()
        
        if dataset_name not in self.DOWNLOADERS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        # create download directory
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        downloader = self.DOWNLOADERS[dataset_name]()
        downloader.download(self.root_dir)
    
    def load_metadata(self, root_dir: Path) -> pd.DataFrame:
        """Load or generate metadata"""
        metadata_path = root_dir / "metadata.csv"
        
        if not metadata_path.exists():
            df = self.metadata_generator.generate(root_dir)
            return df
            
        return pd.read_csv(metadata_path)
    
    def process_dataset(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """Dataset preprocessing pipeline"""
        # Filter data for current split
        split_df = df[df['split'] == split].copy()
        
        if len(split_df) == 0:
            logging.error(f"\nSplit distribution:")
            logging.error(df['split'].value_counts())
            raise ValueError(f"No samples found for {split} split!")
            
        return split_df

    def _split_data(self):
        """Split data into train, validation and test sets"""
        # 기존 split 컬럼이 있다면 사용
        if 'split' in self.metadata.columns:
            train_data = self.metadata[self.metadata['split'] == 'train']
            val_data = self.metadata[self.metadata['split'] == 'val']
            test_data = self.metadata[self.metadata['split'] == 'test']
        else:
            # Split ratio from config
            train_ratio = self.config.dataset.split_ratio.train
            val_ratio = self.config.dataset.split_ratio.val
            
            # Actor ID를 기준으로 split (RAVDESS specific)
            actor_ids = self.metadata['actor_id'].unique()
            np.random.shuffle(actor_ids)
            
            n_actors = len(actor_ids)
            train_actors = actor_ids[:int(n_actors * train_ratio)]
            val_actors = actor_ids[int(n_actors * train_ratio):int(n_actors * (train_ratio + val_ratio))]
            test_actors = actor_ids[int(n_actors * (train_ratio + val_ratio)):]
            
            train_data = self.metadata[self.metadata['actor_id'].isin(train_actors)]
            val_data = self.metadata[self.metadata['actor_id'].isin(val_actors)]
            test_data = self.metadata[self.metadata['actor_id'].isin(test_actors)]
            
            # Save split information
            self.metadata['split'] = 'train'
            self.metadata.loc[self.metadata['actor_id'].isin(val_actors), 'split'] = 'val'
            self.metadata.loc[self.metadata['actor_id'].isin(test_actors), 'split'] = 'test'
            self.metadata.to_csv(self.root_dir / "metadata.csv", index=False)
        
        logging.info(f"\nData split sizes:")
        logging.info(f"Train: {len(train_data)} ({len(train_data)/len(self.metadata):.1%})")
        logging.info(f"Val: {len(val_data)} ({len(val_data)/len(self.metadata):.1%})")
        logging.info(f"Test: {len(test_data)} ({len(test_data)/len(self.metadata):.1%})")
        
        return train_data, val_data, test_data 