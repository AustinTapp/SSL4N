import glob
from torch.utils.data import Dataset
from monai.transforms import (
    AddChanneld,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRangeD,
    Spacingd,
    SpatialPadd,
)

class NiftiData(Dataset):
    def __init__(self):
        # for image check
        # self.path = "C:\\Users\\Austin Tapp\\Documents\\SSL4N\Data\\Skull_Recon_Tests\\2M_and_6M"

        # for standard training
        self.path = "C:\\Users\\Austin Tapp\\Documents\\SSL4N\Data\\Skull_Recon_Tests\\1M\\test"
        self.image_path = glob.glob(self.path + '\\*')

        self.prediction_transform = Compose(

            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes='RAI'),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                ScaleIntensityRangeD(keys=["image"], a_min=-500, a_max=3000, b_min=0, b_max=1),
                CropForegroundd(keys=["image"], source_key="image"),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
                Resized(keys=["image"], spatial_size=(96, 96, 96))
            ]
        )

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        single_path = self.image_path[index]
        image = {"image": single_path}
        return self.prediction_transform(image)
