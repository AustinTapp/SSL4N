import os
import nibabel as nb
import shutil

if __name__ == '__main__':
    data_dir = "D:\\Data\\Brain\\OASIS\\Oasis"
    img_dest_dir = "D:\\Data\\Brain\\OASIS\\Images"
    seg_dest_dir = "D:\\Data\\Brain\\OASIS\\Segs"

    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(("_t88_gfc.img", "_t88_gfc.hdr")):
                file = os.path.join(root, name)
                shutil.copy(file, img_dest_dir)

    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(("fseg.img", "fseg.hdr")):
                file = os.path.join(root, name)
                shutil.copy(file, seg_dest_dir)

