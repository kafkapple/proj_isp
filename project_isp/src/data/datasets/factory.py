from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.data.datasets.ravdess import RavdessDataset

class DataFactory:
    """데이터셋과 데이터로더 생성을 위한 팩토리 클래스"""
    
    DEFAULT_DATALOADER_CONFIG = {
        'shuffle': True,  # train에서만 True로 설정됨
        'drop_last': True,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2
    }
    
    @staticmethod
    def create_dataset_and_loaders(
        config: DictConfig
    ) -> Tuple[Dict[str, Any], Dict[str, DataLoader]]:
        """데이터셋과 데이터 로더를 생성"""
        datasets = {}
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            dataset = DataFactory._create_dataset(config, split)
            if dataset is not None:
                datasets[split] = dataset
                loaders[split] = DataFactory._create_dataloader(dataset, config, split)
                
        return datasets, loaders
    
    @staticmethod
    def _create_dataset(config: DictConfig, split: str):
        """데이터셋 생성"""
        if config.dataset.name == "ravdess":
            return RavdessDataset(config, split)
        raise ValueError(f"Unknown dataset: {config.dataset.name}")
    
    @staticmethod
    def _create_dataloader(dataset, config: DictConfig, split: str):
        """데이터로더 생성"""
        is_train = split == 'train'
        
        # 기본 설정값 사용
        dataloader_config = DataFactory.DEFAULT_DATALOADER_CONFIG.copy()
        
        # validation/test에서는 shuffle 비활성화
        dataloader_config['shuffle'] = is_train and dataloader_config['shuffle']
        dataloader_config['drop_last'] = is_train and dataloader_config['drop_last']
        
        # 설정 파일에 있는 값으로 덮어쓰기
        if hasattr(config.train, 'dataloader'):
            dataloader_config.update({
                'shuffle': is_train and config.train.dataloader.get('shuffle', True),
                'drop_last': is_train and config.train.dataloader.get('drop_last', True),
                'pin_memory': config.train.dataloader.get('pin_memory', True),
                'persistent_workers': config.train.dataloader.get('persistent_workers', True),
                'prefetch_factor': config.train.dataloader.get('prefetch_factor', 2)
            })
        
        # num_workers는 필수 설정값
        num_workers = getattr(config.train, 'num_workers', 4)
        
        return DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            num_workers=num_workers,
            **{k: v for k, v in dataloader_config.items()}
        ) 