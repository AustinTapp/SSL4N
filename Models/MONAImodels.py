import torch.nn as nn
import torch

from monai.networks.nets import ViTAutoEnc
from monai.networks.nets import SwinUNETR


class ViTA(nn.Module):
    def __init__(self, in_channels: int = None, img_size=(None, None, None), patch_size=(None,None,None), out_channels: int = None):
        super().__init__()
        ViTrans = ViTAutoEnc(in_channels=in_channels,
                             img_size=img_size,
                             patch_size=patch_size,
                             out_channels=out_channels,
                             deconv_chns=16,
                             hidden_size=768,
                             mlp_dim=3072,
                             num_layers=12,
                             num_heads=12,
                             pos_embed='conv',
                             dropout_rate=0.0,
                             spatial_dims=3
                             )
        del(ViTrans)


class sUNETR(nn.Module):
    def __init__(self, img_size=None, in_channels: int = None, out_channels: int = None):
        super().__init__()
        Unet = SwinUNETR(img_size=img_size,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         depths=(2, 2, 2, 2),
                         num_heads=(3, 6, 12, 24),
                         feature_size=24,
                         norm_name='instance',
                         drop_rate=0.0,
                         attn_drop_rate=0.0,
                         dropout_path_rate=0.0,
                         normalize=True,
                         use_checkpoint=False,
                         spatial_dims=3,
                         downsample='merging'
                         )
        del(Unet)


