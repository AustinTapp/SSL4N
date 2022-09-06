import nibabel
import os

if __name__ == '__main__':
    data_dir = "D:\\Data\\SSL4N"

    x = None
    y = None
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(("_t1.nii.gz", "_t1ce.nii.gz", "-T1.nii.gz")):
                file = root+"\\"+name
                x = nibabel.load(file).header.get_data_shape()
                if len(x) > len(y):
                    y = x
                    print(y)

    print("Done!")