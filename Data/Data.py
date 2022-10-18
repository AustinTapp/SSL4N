import os
import glob
from torch.utils.data import Dataset
from monai.data import DataLoader
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityD,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

class NiftiData(Dataset):
    def __init__(self):
        # for image check
        # self.path = os.path.join(os.getcwd()+'\\Images')

        # for standard training
        self.path = 'D:\\Data\\Brain\\OASIS\\Images\\AsNifti'
        self.image_path = sorted(glob.glob(self.path + '\\*'))
        self.seg = 'D:\\Data\\Brain\\OASIS\\Segs\\AsNifti'
        self.seg_path = sorted(glob.glob(self.seg + '\\*'))


        self.transform = Compose(

            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes='RAI'),
                ScaleIntensityD(keys=["image"], minv=0.0, maxv=1.0),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )

    def __len__(self):
        return len(self.image_path)

    def transform_data(self, data_dict: dict):
        return self.transform(data_dict)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        segmentation_path = self.seg_path[index]
        image = {"image": image_path, "label": segmentation_path}
        return self.transform_data(image)

    def get_sample(self):
        return self.image_path

