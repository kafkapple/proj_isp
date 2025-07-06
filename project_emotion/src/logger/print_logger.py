from pathlib import Path
from omegaconf import DictConfig, OmegaConf
class PrintLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiment_data= {}
    def print_info(self):
        print(f'===Project===\n{OmegaConf.to_yaml(self.cfg.project, resolve=True)}')
        print(f'===Logger===\n{OmegaConf.to_yaml(self.cfg.logger, resolve=True)}')
        print(f'===Train===\n{OmegaConf.to_yaml(self.cfg.train, resolve=True)}')
        print(f'===Data===\n{OmegaConf.to_yaml(self.cfg.dataset, resolve=True)}')
        print(f'===Model===\n{OmegaConf.to_yaml(self.cfg.model, resolve=True)}')
        print(f'===Model Manager===\n{OmegaConf.to_yaml(self.cfg.model_manager, resolve=True)}')