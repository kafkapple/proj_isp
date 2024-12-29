from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from .datasets.base import BaseDataset
from .datasets.ravdess import RavdessDataset
from .datasets.fer2013 import FER2013Dataset
from .dataloaders.base import BaseDataLoader, get_collate_fn

class DataFactory:
    @staticmethod
    def create_dataset_and_loader(
        name: str,
        config: Dict[str, Any],
        split: str
    ) -> Tuple[BaseDataset, DataLoader]:
        # Create dataset
        if name == "ravdess":
            dataset = RavdessDataset(config, split)
        elif name == "fer2013":
            dataset = FER2013Dataset(config, split)
        else:
            raise ValueError(f"Unknown dataset: {name}")
            
        # 데이터셋에 맞는 collate_fn 가져오기
        collate_fn = get_collate_fn(name)
        
        loader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=(split == 'train'),
            num_workers=config.train.num_workers,
            collate_fn=collate_fn
        )
        
        return dataset, loader