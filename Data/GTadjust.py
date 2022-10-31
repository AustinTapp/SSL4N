import os
import glob
import SimpleITK as sitk
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityD,
    Spacingd,
    SpatialPadd,
)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class NiftiData():
    def __init__(self, path):
        self.path = path
        self.image_path = sorted(glob.glob(self.path + '\\*'))


        self.prediction_transform = Compose(

            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes='RAI'),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                CropForegroundd(keys=["image"], source_key="image"),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
                Resized(keys=["image"], spatial_size=(96, 96, 96))
            ]
        )

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = {"image": image_path}
        return self.prediction_transform(image)


if __name__ == "__main__":
    predict_path = 'C:\\Users\\pmilab\\Auxil\\SSL4N\\Data\\SSL4N_seg_fine_tune\\Test\\segments'
    image = NiftiData(predict_path)
    for item in image:
        file = item['image'].data.meta['filename_or_obj']
        name = file.split("\\")[-1]
        name = name.split(".")[0]
        name = name + "_true_adjusted.nii.gz"
        output = item['image'].detach().cpu().numpy()[0, :, :, :]
        output = sitk.GetImageFromArray(output)
        sitk.WriteImage(output, str(os.path.join(predict_path, name)))
