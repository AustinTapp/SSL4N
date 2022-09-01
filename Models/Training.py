#from Models.MONAImodels import ViTA
from monai.networks.nets import ViTAutoEnc
from torch.nn import L1Loss
from monai.losses import ContrastiveLoss

from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from pytorch_lightning import LightningModule
import torch

class ViTATrain(LightningModule):
    def __init__(self, in_channels=4, img_size=(4, 1, 96, 96, 96), patch_size=(16, 16, 16), batch_size=4, lr = 1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [
            torch.zeros(self.hparams.img_size), torch.zeros(self.hparams.img_size)]

        self.model = ViTAutoEnc(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            pos_embed='conv',
            hidden_size=768,
            mlp_dim=3072,
        )

        #self.model = ViTA(self.hparams.in_channels, self.hparams.img_size, self.hparams.patch_size)
        self.L1 = L1Loss()
        self.contrast = ContrastiveLoss(batch_size=self.hparams.batch_size, temperature=0.05)

    def forward(self, inputs, inputs2):
        outputs_v1, hidden_v1 = self.model(inputs)
        outputs_v2, hidden_v2 = self.model(inputs2)
        return outputs_v1, outputs_v2

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _prepare_batch(self, batch):
        return batch[0]['image'], batch[0]['image_2'], batch[0]['gt_image']

    def _common_step(self, batch, batch_idx, stage: str):
        inputs, inputs_2, gt_input = self._prepare_batch(batch)

        outputs_v1, outputs_v2, = self.forward(inputs, inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        r_loss = self.L1(outputs_v1, gt_input)

        if stage == 'train':
            import pdb; pdb.set_trace()

        cl_loss = self.contrast(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss
        train_steps = self.current_epoch + batch_idx

        self.log_dict({
            f'{stage}_loss': total_loss,
            'step':  train_steps,
            'epoch': self.current_epoch})

        if train_steps % 100 == 0:
            self.log_dict({
                'L1': r_loss,
                'Contrastive': cl_loss,
                'epoch': self.current_epoch,
                'step': train_steps})
            self.logger.log_image(key="Images", images=[gt_input.cpu().numpy()[0, 0, :, :, 38],
                                                        outputs_v1.cpu().numpy()[0, 0, :, :, 38],
                                                        outputs_v2.cpu().numpy()[0, 0, :, :, 38]],
                                  caption=["GT", "Recon1", "Recon2"])

        return total_loss


