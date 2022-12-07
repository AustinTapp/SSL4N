from monai.networks.nets import ViTAutoEnc
from torch.nn import L1Loss
from monai.losses import ContrastiveLoss
import numpy as np
from pytorch_lightning import LightningModule
import torch

class ViTATrain(LightningModule):
    def __init__(self, in_channels=4, img_size=(1, 1, 96, 96, 96), patch_size=(16, 16, 16), batch_size=1, lr=1e-4,
                 ckpt_path="C:\\Users\\Austin Tapp\\Documents\\SSL4N\\saved_models\\loss\\epoch=19070-step=19070.ckpt"):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [torch.zeros(self.hparams.img_size)]

        self.model = ViTAutoEnc(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            pos_embed='conv',
            hidden_size=768,
            mlp_dim=3072,
        )

    def forward(self, inputs):
        outputs, hiddens = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch['image'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    # index needed here, batch is a list of dicts with size 1,
    def _prepare_batch(self, batch):
        return batch[0]['image']

    def _common_step(self, batch, batch_idx, stage: str):
        inputs = self._prepare_batch(batch)
        outputs, hiddens = self.forward(inputs)
        train_steps = self.current_epoch + batch_idx



