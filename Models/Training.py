import monai.data
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss, DiceLoss
import numpy as np
from pytorch_lightning import LightningModule
import torch

class UNetR_Train(LightningModule):
    def __init__(self, in_channels=4, img_size=(1, 1, 96, 96, 96), patch_size=(16, 16, 16), batch_size=1, lr=1e-5,
                 ckpt_path = "C:\\Users\\pmilab\\Auxil\\SSL4N\\saved_models\\Saved\\T1inf_FT_2_epoch=246-step=248.ckpt"):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [torch.zeros(self.hparams.img_size)]

        self.model = UNETR(in_channels=1,
                     out_channels=4,
                     img_size=(96, 96, 96),
                     feature_size=16,
                     hidden_size=768,
                     mlp_dim=3072,
                     num_heads=12,
                     pos_embed="conv",
                     norm_name="instance",
                     res_block=True,
                     dropout_rate=0.0,
                     )

        self.checkpoint = torch.load(self.hparams.ckpt_path)
        self.vit_weights = self.checkpoint['state_dict']
        self.model_dict = self.model.vit.state_dict()
        self.vit_weights = {k: v for k, v in self.vit_weights.items() if k in self.model_dict}
        self.model_dict.update(self.vit_weights)
        self.model.vit.load_state_dict(self.model_dict)
        del self.model_dict, self.vit_weights, self.checkpoint
        print('Pretrained Weights Succesfully Loaded !')

        self.DSCE_Loss = DiceCELoss(to_onehot_y=False, softmax=True)
        self.DSC_Loss = DiceLoss(include_background=False)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     img = self._prepare_batch(batch)
    #     return self.forward(img)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        return optimizer

    def _prepare_batch(self, batch):
        return batch[0]['image'], batch[0]['label']

    def _common_step(self, batch, batch_idx, stage: str):
        inputs, gt_input = self._prepare_batch(batch)

        outputs = self.forward(inputs)
        DSCE_loss = self.DSCE_Loss(outputs, gt_input)
        DSC = 1 - (self.DSC_Loss(outputs, gt_input))
        train_steps = self.current_epoch + batch_idx
        #(outputs.detach().cpu().numpy().argmax(1)) prediction output for 4 labels

        self.log_dict({
            f'{stage}_loss': DSCE_loss.item(),
            f'{stage}_DSC': DSC.item(),
            'step': float(train_steps),
            'epoch': float(self.current_epoch)}, batch_size=self.hparams.batch_size)

        if train_steps % 100 == 0:
            self.log_dict({
                'DSCE_Loss': DSCE_loss.item(),
                'epoch': float(self.current_epoch),
                'step': float(train_steps)}, batch_size=self.hparams.batch_size)
            self.logger.log_image(key="Ground Truth", images=[
                ((gt_input.detach().cpu().numpy().argmax(1))*85)[0, :, :, 38]],
                caption=["GT Segmentations"])
            self.logger.log_image(key="Input Image", images=[
                (inputs.detach().cpu().numpy()*255)[0, 0, :, :, 38]],
                caption=["Input Image"])
            self.logger.log_image(key="Prediction", images=[
                ((outputs.detach().cpu().numpy().argmax(1))*85)[0, :, :, 38]],
                caption=["Segmentations "])

        return DSCE_loss


