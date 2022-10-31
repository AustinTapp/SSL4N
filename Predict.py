import os
import SimpleITK as sitk
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Data.Dataloader import MRIdata
from Models.Training import UNetR_Train

from pytorch_lightning import Trainer

if __name__ == "__main__":
    checkpoint_path = "C:\\Users\\pmilab\\Auxil\\SSL4N\\saved_models\\loss\\epoch=377-step=377.ckpt"
    predict_path = "C:\\Users\\pmilab\\Auxil\\SSL4N\\Data\\SSL4N_seg_fine_tune\\Test"

    trainer = Trainer(accelerator="gpu", devices=[0])

    UNETR = UNetR_Train().load_from_checkpoint(checkpoint_path)
    predict = trainer.predict(UNETR, datamodule=MRIdata(batch_size=1))

    for predictions in predict:
        file = predictions.data.meta['filename_or_obj']
        name = file.split("\\")[-1]
        name = name.split(".")[0]
        name = name + "_predict.nii.gz"
        output = predictions.detach().cpu().numpy().argmax(1)[0, :, :, :]
        output = sitk.GetImageFromArray(output)
        sitk.WriteImage(output, str(os.path.join(predict_path, name)))

