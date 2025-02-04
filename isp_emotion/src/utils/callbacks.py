from pytorch_lightning.callbacks import TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._current_epoch = 1
    
    def setup(self, trainer, pl_module, stage=None):
        super().setup(trainer, pl_module, stage)
        if stage == "fit":
            self._current_epoch = 1
    
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        # Epoch을 1부터 시작하도록 수정
        total_batches = self.total_train_batches
        prefix = f"Epoch {self._current_epoch}"
        self.train_progress_bar.set_description(prefix)
    
    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        prefix = f"Validation Epoch {self._current_epoch}"
        self.val_progress_bar.set_description(prefix)
    
    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        self._current_epoch += 1 