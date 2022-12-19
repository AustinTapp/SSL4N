import os
import glob
from torch.utils.data import Dataset
from monai.transforms import (
    LoadImaged,
    OrientationD,
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRangeD,
    ResizeD,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    SpatialPadD,
    ThresholdIntensityD
)

class NiftiData(Dataset):
    def __init__(self):
        # for image check
        # self.path = "C:\\Users\\Austin Tapp\\Documents\\SSL4N\Data\\Skull_Recon_Tests\\2M_and_6M"

        # for standard training, !!change this path!!!
        self.path = "C:\\Users\\pmilab\\Auxillary\\SSL4N\\Data\\asNifti_nobed"
        self.image_path = glob.glob(self.path + '\\*')

        self.transform = Compose(

            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                OrientationD(keys=["image"], axcodes='RAI'),
                Spacingd(keys=["image"], pixdim=(
                    1.0, 1.0, 1.0), mode=("bilinear")),
                # segmentation change to nearest
                #CropForegroundd(keys=["image"], source_key="image"),
                ResizeD(keys=["image"], spatial_size=(256, 256, 128)),
                SpatialPadD(keys=["image"], spatial_size=(256, 256, 256)),
                RandSpatialCropSamplesd(keys=["image"], roi_size=(64, 64, 64), random_size=False, num_samples=4),
                CopyItemsd(keys=["image"], times=3, names=["image_2", "gt_image", "gt_mask"], allow_missing_keys=False),
                ThresholdIntensityD(keys=["gt_mask"], threshold=200),
                ScaleIntensityRangeD(keys=["image", "image_2", "gt_image"], a_min=-500, a_max=3000, b_min=0, b_max=1),
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
                RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),

            ]
        )


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        single_path = self.image_path[index]
        image = {"image": single_path}
        return self.transform(image)

    def get_sample(self):
        return self.image_path

