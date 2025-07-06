import hydra
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.utils.config import filter_config, setup_output_dir
from src.data.processor import DataProcessor
from src.model.factory import ModelFactory
from src.utils.callbacks import CustomProgressBar

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.general.seed)
    
    # output directory setup
    output_dir = setup_output_dir(cfg)
    
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
    
    # local logging
    logger.experiment.log_artifact(
        str(output_dir),
        name="experiment_outputs",
        type="outputs"
    )
    
    # data and model setup    
    data_processor = DataProcessor(cfg)
    loaders = data_processor.create_dataloaders()
    model = ModelFactory.create(cfg)  # LightningModule 반환

    # checkpoint callback
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

    # trainer setup
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=[checkpoint_callback, CustomProgressBar()],
        logger=logger,
        fast_dev_run=cfg.debug.fast_dev_run,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.get('accumulate_grad_batches', 1),
        gradient_clip_val=cfg.train.get('gradient_clip_val', None),
        log_every_n_steps=cfg.logging.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=cfg.logging.show_model_summary,
        default_root_dir=cfg.dirs.outputs,
        num_sanity_val_steps=0,
    )
    
    # training
    trainer.fit(model, loaders["train"], loaders["val"])
    if 'test' in loaders:
        trainer.test(model, loaders['test'])

    wandb.finish()

if __name__ == "__main__":
    main()