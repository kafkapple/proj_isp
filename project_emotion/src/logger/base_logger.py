from omegaconf import DictConfig
from pathlib import Path
import os
from .wandb_logger import WandbLogger
from .print_logger import PrintLogger
class Logger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.setup_exp_dirs()
        self.setup_loggers()
    def setup_exp_dirs(self):
        for subdir in self.cfg.dirs.subdirs:
            os.makedirs(Path(self.cfg.dirs.outputs) / subdir, exist_ok=True)

    def setup_loggers(self):
        self.wandb_logger = WandbLogger(self.cfg)
        self.print_logger = PrintLogger(self.cfg)
