from Data.Dataloader import ADNIdata


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
        # overfit_batches=1,
        log_every_n_steps=1,
    )
    # effective batch size is double due to reversing
    size = 128
    pixdim = 1.0
    trainer.fit(
        model=FeatureLearner(lr=1e-3, mask_weight=3.0, img_size=size, pixdim=pixdim, lam=1.0),
        datamodule=FeatureDataModule(
            batch_size=3,
            img_size=size,
            pixdim=pixdim),
        # ckpt_path = '/home/abhi/Code/NST3DBrain/saved_models/loss/epoch=3-step=4.ckpt'
    )