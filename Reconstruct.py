import os
import SimpleITK as sitk
import monai.data
import numpy
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Data.Dataloader import CTdata
from Models.Training import ViTATrain
from pytorch_lightning import Trainer

if __name__ == "__main__":
    checkpoint_path = "C:\\Users\\Austin Tapp\\Documents\\SSL4N\\saved_models\\loss\\epoch=19070-step=19070.ckpt"
    prediction_output_path = "C:\\Users\\Austin Tapp\\Documents\\SSL4N\\Recon\\1M"

    trainer = Trainer(accelerator="gpu", devices=[0])

    ViT = ViTATrain().load_from_checkpoint(checkpoint_path)
    print("Training weights successfully loaded!")

    predict = trainer.predict(ViT, datamodule=CTdata(batch_size=1))

    for predictions in predict:
        outputs = predictions.to(dtype=torch.float32)
        output_array = numpy.clip(outputs.detach().cpu().numpy(), 0, 1)
        output = (output_array * 255)[0, 0, :, :, :]
        output = numpy.transpose(output)
        output = numpy.flip(output)
        output = sitk.GetImageFromArray(output)

        file = predictions.data.meta['filename_or_obj']
        name = file.split("\\")[-1]
        name = name.split(".")[0]
        name = name + "_predict.nii.gz"
        print("Writing:", name)

        sitk.WriteImage(output, str(os.path.join(prediction_output_path, name)))

    #TODO
    #monai.transforms.SaveImageD - return to original