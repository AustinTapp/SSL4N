import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Data.Dataloader import MRIdata
from Models.Training import UNetR_Train

from pytorch_lightning import Trainer

if __name__ == "__main__":
    checkpoint_path = "some_path"

    trainer = Trainer(accelerator="gpu", devices=[0])

    UNETR = UNetR_Train()
    UNETR.load_from_checkpoint(checkpoint_path)
    prediction = trainer.predict(model=UNETR, datamodule=MRIdata(batch_size=1))
