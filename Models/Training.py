#from Models.MONAImodels import ViTA
from monai.networks.nets import ViTAutoEnc
from torch.nn import L1Loss
from monai.losses import ContrastiveLoss, SSIMLoss
from monai.losses.spatial_mask import MaskedLoss
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

class ViTATrain(LightningModule):
    def __init__(self, in_channels=4, mask_weight=3, img_size=(1, 1, 64, 64, 64), patch_size=(16, 16, 16), batch_size=1, lr=1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [
            torch.zeros(self.hparams.img_size), torch.zeros(self.hparams.img_size)]

        self.model = ViTAutoEnc(
            in_channels=1,
            img_size=(64, 64, 64),
            patch_size=(16, 16, 16),
            pos_embed='conv',
            hidden_size=768,
            mlp_dim=3072,
        )

        #self.model = ViTA(self.hparams.in_channels, self.hparams.img_size, self.hparams.patch_size)
        self.L1 = L1Loss()
        self.Contrast = ContrastiveLoss(temperature=0.05)
        self.SSIM = SSIMLoss(spatial_dims=3)
        self.Mask = MaskedLoss(loss=F.l1_loss)


    def forward(self, inputs, inputs2):
        outputs_v1, hidden_v1 = self.model(inputs)
        outputs_v2, hidden_v2 = self.model(inputs2)
        return outputs_v1, outputs_v2

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, cooldown=5),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    # index needed here, batch is a list of dicts with size 1,
    def _prepare_batch(self, batch):
        image_batch = torch.cat([batch[i]['image'] for i in range(len(batch))])
        image_2_batch = torch.cat([batch[i]['image_2'] for i in range(len(batch))])
        gt_batch = torch.cat([batch[i]['gt_image'] for i in range(len(batch))])
        gt_mask = torch.cat([batch[i]['gt_mask'] for i in range(len(batch))])
        return image_batch, image_2_batch, gt_batch, gt_mask

    def _common_step(self, batch, batch_idx, stage: str):
        inputs, inputs_2, gt_input, gt_mask = self._prepare_batch(batch)

        outputs_v1, outputs_v2 = self.forward(inputs, inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        def get_mask(img, weight):
            mask = torch.ones_like(img)
            mask[img > 0.01] = weight
            return mask

        gt_input_mask = get_mask(img=gt_mask, weight=self.hparams.mask_weight)
        outputs_v1_mask = get_mask(img=outputs_v1, weight=self.hparams.mask_weight)
        outputs_v2_mask = get_mask(img=outputs_v2, weight=self.hparams.mask_weight)

        skull_mask1 = gt_input_mask + outputs_v1_mask
        skull_mask2 = gt_input_mask + outputs_v2_mask

        r_loss = self.L1(outputs_v1, gt_input)
        cl_loss = self.Contrast(flat_out_v1, flat_out_v2)
        ssim_loss = self.SSIM(outputs_v2, gt_input, data_range=inputs.max().unsqueeze(0))

        mask_loss1 = self.Mask(outputs_v1, gt_input, mask=skull_mask1)
        mask_loss2 = self.Mask(outputs_v2, gt_input, mask=skull_mask2)
        mask_loss = (mask_loss1+mask_loss2)/2

        # Adjust the CL loss by Recon Loss
        total_loss = mask_loss + ssim_loss + r_loss + cl_loss * r_loss
        train_steps = self.current_epoch + batch_idx

        self.log_dict({
            f'{stage}_loss': total_loss.item(),
            'step': float(train_steps),
            'epoch': float(self.current_epoch)}, batch_size=self.hparams.batch_size)

        if train_steps % 10 == 0:
            self.log_dict({
                'L1': r_loss.item(),
                'Contrastive': cl_loss.item(),
                'SSIM': ssim_loss.item(),
                'Skull Mask': mask_loss.item(),
                'epoch': float(self.current_epoch),
                'step': float(train_steps)}, batch_size=self.hparams.batch_size)

            self.logger.log_image(key="Ground Truth", images=[
                (gt_input.detach().cpu().numpy()*255)[0, 0, :, :, 32],
                (gt_input_mask.detach().cpu().numpy()*50)[0, 0, :, :, 32]],
                caption=["GT", "GT Mask"])
            self.logger.log_image(key="Input (Transformed) Images", images=[
                (inputs.detach().cpu().numpy()*255)[0, 0, :, :, 32],
                (inputs_2.detach().cpu().numpy()*255)[0, 0, :, :, 32]],
                caption=["Input 1", "Input 2"])
            outputs_v1 = outputs_v1.to(dtype=torch.float16)
            outputs_v2 = outputs_v2.to(dtype=torch.float16)
            outputs_v1_array = np.clip(outputs_v1.detach().cpu().numpy(), 0, 1)
            outputs_v2_array = np.clip(outputs_v2.detach().cpu().numpy(), 0, 1)
            self.logger.log_image(key="Reconstructed Images", images=[
                (outputs_v1_array*255)[0, 0, :, :, 32],
                (outputs_v2_array*255)[0, 0, :, :, 32],
                (outputs_v1_mask.detach().cpu().numpy()*25)[0, 0, :, :, 32],
                (outputs_v2_mask.detach().cpu().numpy()*25)[0, 0, :, :, 32]],
                caption=["Recon1", "Recon2", "ReconMask1", "ReconMask2"])

        return total_loss


