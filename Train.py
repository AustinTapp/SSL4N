from Data.Dataloader import MRIdata
from Models.Training import ViTATrain

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="SSL4N")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="./saved_models/loss/", save_top_k=1, monitor="val_loss")
    trainer = Trainer(
        logger=wandb_logger,
        gpus=[0],
        max_epochs=500,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
    )
    size = 128
    dims = 1
    trainer.fit(
        model=ViTATrain(),
        datamodule=MRIdata(
            batch_size=3,
            img_size=size,
            dimensions=dims)
    )


