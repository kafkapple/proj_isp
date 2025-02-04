import hydra
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.data.datasets.factory import DataFactory
from src.model.factory import ModelFactory
import wandb
from pathlib import Path

def filter_config(cfg):
    """Extract specific parts of Hydra config."""
    selected_keys = ["train", "model", "dataset"]  # 원하는 config key
    return {key: OmegaConf.to_container(cfg[key], resolve=True) for key in selected_keys if key in cfg}

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.general.seed)
    
    # 출력 디렉토리 생성
    output_dir = Path(cfg.dirs.outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in cfg.dirs.subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # wandb logger
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        save_dir=cfg.wandb.save_dir,
        version=cfg.general.timestamp,
        log_model=False,
        save_code=True,
        config=filter_config(cfg)
    )
    
    # 로컬에도 로깅 데이터 저장 (디렉토리 생성 후)
    logger.experiment.log_artifact(
        str(output_dir),  # Path 객체를 문자열로 변환
        name="experiment_outputs",
        type="outputs"
    )
            
    datasets, loaders = DataFactory.create_dataset_and_loaders(cfg)
    print("done")
    model = ModelFactory.create(cfg)

    # 체크포인트 콜백
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.dirs.outputs}/checkpoints",
        filename="best-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        save_weights_only=True,
    )

    # Trainer 설정 구성
    trainer_config = {
        'max_epochs': cfg.train.max_epochs,
        'callbacks': [checkpoint_callback],
        'logger': logger,
        'fast_dev_run': cfg.debug.fast_dev_run,
        
        # 하드웨어/성능 설정
        'accelerator': cfg.train.accelerator,
        'devices': cfg.train.devices,
        'precision': cfg.train.precision,
        
        # 옵티마이저 관련
        'accumulate_grad_batches': cfg.train.get('accumulate_grad_batches', 1),
        'gradient_clip_val': cfg.train.get('gradient_clip_val', None),
        
        # 로깅 관련
        'log_every_n_steps': cfg.logging.log_every_n_steps,
        
        # Early stopping 관련
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': cfg.logging.show_model_summary,
    }

    trainer = pl.Trainer(**trainer_config)
    
    trainer.fit(model, loaders["train"], loaders["val"])
    if 'test' in loaders:
        trainer.test(model, loaders['test'])

    wandb.finish()

if __name__ == "__main__":
    main()