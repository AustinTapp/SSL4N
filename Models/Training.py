#from Models.MONAImodels import ViTA
from monai.networks.nets import ViTAutoEnc
from torch.nn import L1Loss
from monai.losses import ContrastiveLoss

from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from pytorch_lightning import LightningModule

import torch

class ViTATrain(LightningModule):
    def __init__(self, in_channels=4, img_size=(96, 96, 96), patch_size=(16, 16, 16), batch_size=4):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [
            torch.zeros(self.hparams.img_size)]

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
        self.contrast = ContrastiveLoss(batch_size=self.hparams.batch_size*2, temperature=0.05)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _prepare_batch(self, batch):
        return batch['image'], batch['image_2'], batch['gt_image']

    def _common_step(self, batch, batch_idx, stage: str):
        inputs, inputs_2, gt_input = self._prepare_batch(batch)

        outputs_v1, hidden_v1 = self.model(inputs)
        outputs_v2, hidden_v2 = self.model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        r_loss = self.L1(outputs_v1, gt_input)
        cl_loss = self.contrast(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss

        self.log({
            f'{stage}_loss': total_loss,
            'step':  self.train_steps,
            'epoch': self.current_epoch})

#need to finish logging
        if self.train_steps % 100 == 0:
            self.log({
                'learning rate': self.model.optimizers[0].param_groups[0]['lr'],
                'L1': r_loss,
                'epoch': self.current_epoch,
                'step': self.train_steps,
                'Image': wandb.Image(model.get_current_visuals()['real_A'].squeeze().data.cpu().numpy()[:, :, 32]),
                'Labels': {
                    'true': wandb.Image(
                        model.get_current_visuals()['rec_A'].squeeze().data.cpu().numpy()[:, :, 32]),  # recreated
                    'pred': wandb.Image(
                        model.get_current_visuals()['fake_B'].squeeze().data.cpu().numpy()[:, :, 32]),  # t2 for now
                },
                'Masks': {'true': wandb.Image(
                    model.get_current_visuals()['skull_mask_true'].squeeze().data.cpu().numpy()[:, :, 32]),
                    'pred': wandb.Image(
                        model.get_current_visuals()['skull_mask_pred'].squeeze().data.cpu().numpy()[:, :, 32])}
            })  # end

        if total_steps % opt.print_freq == 0:
            losses = model.get_current_losses()
            t = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
            mae = MeanAbsoluteError()
            val_score = mae(model.get_current_visuals()['rec_A'].cpu(), model.get_current_visuals()['real_A'].cpu())
            # edit this


        return total_loss


