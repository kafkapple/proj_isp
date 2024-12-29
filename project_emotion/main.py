import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import logging

from src.logger.base_logger import Logger
from src.data.factory import DataFactory
from src.models.factory import ModelFactory
from src.utils.model_manager import ModelManager

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def train(config: DictConfig):
    logger = Logger(config)
    logger.print_logger.print_info()
 
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=config.logger.wandb.project_name,
        entity=config.logger.wandb.entity,
        tags=config.logger.wandb.tags,
        save_dir=config.dirs.outputs
    )
    
    # Prepare datasets and dataloaders
    train_dataset, train_loader = DataFactory.create_dataset_and_loader(
        config.dataset.name,
        config,
        split="train"
    )
    val_dataset, val_loader = DataFactory.create_dataset_and_loader(
        config.dataset.name,
        config,
        split="val"
    )
    test_dataset, test_loader = DataFactory.create_dataset_and_loader(
        config.dataset.name,
        config,
        split="test"
    )
       # Initialize model manager
    model_manager = ModelManager(config.model_manager.base_path)
    
    # Initialize model
    model = ModelFactory.create(config.model.name, config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{config.dirs.outputs}/checkpoints/{config.model.name}",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        accelerator="auto",
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    test_results = trainer.test(model, test_loader)[0]
    logging.info(f"\nTest Results:\n{test_results}")

if __name__ == "__main__":
    train()