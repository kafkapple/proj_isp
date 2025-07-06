from pathlib import Path
import pandas as pd
import librosa
import logging
from typing import Dict, List
from omegaconf import DictConfig

class MetadataGenerator:
    def __init__(self, config: DictConfig):
        self.config = config
        # sample_rate를 선택적으로 가져오기 (기본값: None)
        self.sample_rate = config.dataset.get('sample_rate', None)
        
    def generate(self, root_dir: Path) -> pd.DataFrame:
        """Generate metadata for audio files"""
        metadata = []
        
        # Actor 폴더별로 순회
        for actor_dir in root_dir.glob("Actor_*"):
            if not actor_dir.is_dir():
                continue
            
            # 각 Actor 폴더 내의 wav 파일 처리
            for audio_path in actor_dir.glob("*.wav"):
                try:
                    file_info = self._process_audio_file(audio_path)
                    if file_info:
                        metadata.append(file_info)
                except Exception as e:
                    logging.warning(f"Error processing file {audio_path}: {e}")
                    continue
        
        if not metadata:
            raise ValueError(f"No valid files found in {root_dir}")
                
        return self._create_dataframe(metadata)
    
    def _process_audio_file(self, file_path: Path) -> Dict:
        """Process single audio file and extract metadata"""
        # RAVDESS filename format: 03-01-04-01-02-01-12.wav
        # modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        filename = file_path.stem
        parts = filename.split("-")
        
        if len(parts) != 7:
            logging.warning(f"Invalid filename format: {filename}")
            return None
            
        # Load audio for duration
        audio, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        return {
            "file_path": str(file_path),
            "emotion": self.config.dataset.class_names[int(parts[2]) - 1],
            "intensity": "normal" if parts[3] == "01" else "strong",
            "actor_id": parts[6],
            "duration": duration,
            "sample_rate": sr
        }
    
    def _create_dataframe(self, metadata: List[Dict]) -> pd.DataFrame:
        """Create and process metadata DataFrame"""
        if not metadata:
            raise ValueError("No valid files found")
            
        df = pd.DataFrame(metadata)
        
        # Add split column (train/val/test)
        df["split"] = "train"  # default
        
        # Use specific actors for validation and test
        val_actors = self.config.dataset.get("val_actors", ["15", "16"])
        test_actors = self.config.dataset.get("test_actors", ["17", "18"])
        
        df.loc[df["actor_id"].isin(val_actors), "split"] = "val"
        df.loc[df["actor_id"].isin(test_actors), "split"] = "test"
        
        logging.info("\nDataset split distribution:")
        logging.info(df["split"].value_counts())
        
        return df 