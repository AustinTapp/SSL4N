import torch.nn as nn
from monai.networks.nets import ViTAutoEnc
from monai.networks.nets import UNETR


class ViTA(nn.Module):
    def __init__(self, in_channels: int = None, img_size=(None, None, None), patch_size=(None,None,None)):
        super().__init__()
        self.ViTrans = ViTAutoEnc(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                pos_embed='conv',
                hidden_size=768,
                mlp_dim=3072,
    )

    def forward(self, images):
        return self.ViTrans(images)


class SegNet(nn.Module):
    def __init__(self, img_size=(None, None, None), in_channels: int = None, out_channels: int = None):

        super().__init__()
        self.Unetr = UNETR(in_channels=1,
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



#need to add forward stuff
