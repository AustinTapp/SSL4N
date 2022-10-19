import os
import glob

import torch
from torch.utils.data import Dataset
from monai.data import DataLoader
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropd,
    RandShiftIntensityd,
    ScaleIntensityD,
    Spacingd,
    SpatialPadd,
    RandSpatialCropSamplesd,
    RandRotate90d,
)

class NiftiData(Dataset):
    def __init__(self):
        # for image check
        # self.path = os.path.join(os.getcwd()+'\\Images')

        # for standard training
        self.path = 'C:\\Users\\pmilab\\Auxil\\SSL4N\\Data\\SSL4N_seg_initial\\ImagesAsNifti'
        self.image_path = sorted(glob.glob(self.path + '\\*'))
        self.seg = 'C:\\Users\\pmilab\\Auxil\\SSL4N\\Data\\SSL4N_seg_initial\\SegsAsNifti'
        self.seg_path = sorted(glob.glob(self.seg + '\\*'))


        self.transform = Compose(

            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes='RAI'),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                ScaleIntensityD(keys=["image"], minv=0.0, maxv=1.0),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
                RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(96, 96, 96), random_size=False, num_samples=1),
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
        image_transformed = self.transform_data(image)
        labels = []
        for i in range(4):
            zeros = torch.zeros_like(image_transformed[0]["label"])
            zeros[image_transformed[0]["label"] == i] = 1
            labels.append(zeros)
        modified_label = torch.stack(labels, dim=1)
        image_transformed[0]["label"] = torch.squeeze(modified_label, 0)
        return image_transformed

    def get_sample(self):
        return self.image_path

