import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Data.Dataloader import MRIdata
from Models.Training import UNetR_Train

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="SSL4N")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="./saved_models/loss/", save_top_k=1, monitor="val_loss", save_on_train_epoch_end=True)
    checkpoint_path = "C:\\Users\Austin Tapp\\Documents\\SSL4N\\saved_models\\loss\\All_but_CHLAs_ epoch=195-step=261.ckpt"

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=[0],
        max_epochs=500,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
        vit_ckpt_path=checkpoint_path,
    )

    trainer.fit(
        model=UNetR_Train(),
        datamodule=MRIdata(
            batch_size=1)
    )

#change to fine tune for segmentaitons (ALZ)