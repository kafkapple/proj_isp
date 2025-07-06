import wandb
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
class WandbLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_name = cfg.logger.wandb.project_name
        self.entity = cfg.logger.wandb.entity
        self.run = self.init_wandb()
        
    def init_wandb(self):
        selected_config = OmegaConf.select(self.cfg, 'defaults,train')
        try:
            run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=self.cfg.project.timestamp,
                config=OmegaConf.to_container(self.cfg.train, resolve=True), # run column info
                tags=self.cfg.logger.wandb.tags
            )
        except Exception as e:
            print(f'Failed to initialize WandB: {e}')
            return None
        return run
    def log_params(self, params: dict):
        wandb.config.update(params)
    def log_metrics(self, metrics: dict, step=None):
        wandb.log(metrics, step=step)
    
    def add_summary(self, summary_dict: dict):
        wandb.summary.update(summary_dict)
  
    def save_model(self, model, filename=None):
        if filename is None:
            filename = f'{self.run_name}.pt'
        filepath = Path(self.save_dir) / filename
        torch.save(model.state_dict(), filepath)
        wandb.save(filepath)
        print(f'Model saved to {filepath}')
    def add_artifact(self, artifact_name: str, artifact_type: str, artifact_path: str):
        wandb.log_artifact(artifact_name, artifact_type, artifact_path)
    def finish(self):
        wandb.finish()
