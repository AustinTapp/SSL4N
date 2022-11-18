import os
import shutil
import SimpleITK as sitk

if __name__ == '__main__':
    data_dir = "D:\\Data\\Skull\\NormalCases_All"
    list_subfolders_with_paths = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for i in range(len(list_subfolders_with_paths)):
        nifti_folder = os.path.join(list_subfolders_with_paths[i], "asNifti")
        #print(nifti_folder)
        isExist = os.path.exists(nifti_folder)
        if not isExist:
            os.makedirs(nifti_folder)
        patient_subfolder_with_path = [f.path for f in os.scandir(list_subfolders_with_paths[i]) if f.is_dir()]
        for j in range(len(patient_subfolder_with_path)):
            #print(patient_subfolder_with_path[j].split('\\')[-1])
            patient_imagefiles_with_path = [f.path for f in os.scandir(patient_subfolder_with_path[j]) if f.is_dir()]
            for k in range(len(patient_imagefiles_with_path)):
                z = 0
                #print(patient_imagefiles_with_path[k])
                if patient_imagefiles_with_path[k].endswith('Old'):
                    shutil.rmtree(patient_imagefiles_with_path[k])
                else:
                    z = z+1
                    DCMfiles_with_path = [f.path for f in os.scandir(patient_imagefiles_with_path[k]) if f.is_dir()]
                    #print(DCMfiles_with_path[0])

                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(DCMfiles_with_path[0])
                    reader.SetFileNames(dicom_names)
                    dicom_image = reader.Execute()
                    nifti_image_name = os.path.join(data_dir,
                                       list_subfolders_with_paths[i].split('\\')[-1], "asNifti",
                                       patient_subfolder_with_path[j].split('\\')[-1] + "_" +
                                       patient_imagefiles_with_path[k].split('\\')[-1] + "_.nii.gz")
                    #
                    sitk.WriteImage(dicom_image, nifti_image_name)
                    print(f"Writing {nifti_image_name} to file was successful...\n")



        # for name in files:
        #     if name.endswith(("_t2.nii.gz", "_T2w.nii.gz", "-T2.nii.gz")):
        #         file = root+"\\"+name
        #         #print(file)
        #         shutil.copy(file, dest_dir)

    print("Done!")