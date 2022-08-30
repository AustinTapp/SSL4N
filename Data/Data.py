import os
import glob
from torch.utils.data import Dataset
from monai.data import DataLoader
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled
)

class NiftiData(Dataset):
    def __init__(self):
        self.path = os.path.join(os.getcwd()+'\\Images')
        self.images = glob.glob(self.path + '\\*')

        self.transform = Compose(

            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    2.0, 2.0, 2.0), mode=("bilinear")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
                RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
                CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
                OneOf(transforms=[
                    RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                                       max_spatial_size=32),
                    RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                                       max_spatial_size=64),
                ]
                ),
                RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
                OneOf(transforms=[
                    RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                                       max_spatial_size=32),
                    RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                                       max_spatial_size=64),
                ]
                ),
                RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8)
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = self.images[item]
        image = {"image": path}
        return self.transform(image)

    def get_sample(self):
        return self.images
