import os
import shutil
import SimpleITK as sitk

if __name__ == '__main__':
    data_dir = "C:\\Users\\Austin Tapp\\Downloads\\CQ500"
    list_subfolders_with_paths = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for i in range(len(list_subfolders_with_paths)):
        nifti_folder = os.path.join(data_dir, "asNifti")
        #print(nifti_folder)
        isExist = os.path.exists(nifti_folder)
        if not isExist:
            os.makedirs(nifti_folder)
        patient_subfolder_with_path = [f.path for f in os.scandir(list_subfolders_with_paths[i]) if f.is_dir()]

        for j in range(len(patient_subfolder_with_path)):
            #print(patient_subfolder_with_path[j].split('\\')[-1])
            patient_scanfiles_with_path = [f.path for f in os.scandir(patient_subfolder_with_path[j]) if f.is_dir()]

            for k in range(len(patient_scanfiles_with_path)):
                # print(patient_subfolder_with_path[j].split('\\')[-1])
                patient_imagefiles_with_path = [f.path for f in os.scandir(patient_scanfiles_with_path[k]) if f.is_dir()]

                for l in range(len(patient_imagefiles_with_path)):
                    #print(patient_imagefiles_with_path[k])
                    if patient_imagefiles_with_path[k].endswith('Old'):
                        shutil.rmtree(patient_imagefiles_with_path[k])
                    #elif patient_imagefiles_with_path[k].endswith('asNifti'):
                    #    shutil.rmtree(patient_imagefiles_with_path[k])
                    else:
                        max_size = 0
                        DCM_used_path = None
                        DCMfiles_path = [f.path for f in os.scandir(patient_imagefiles_with_path[l]) if f.is_dir()]
                        #print(DCMfiles_path[0])
                        for folder in DCMfiles_path:
                            size = os.stat(folder).st_size
                            if size > max_size:
                                max_size = size
                                DCM_used_path = folder
                        reader = sitk.ImageSeriesReader()
                        dicom_names = reader.GetGDCMSeriesFileNames(DCM_used_path)
                        reader.SetFileNames(dicom_names)
                        dicom_image = reader.Execute()
                        nifti_image_name = os.path.join(data_dir, "asNifti",
                                           patient_subfolder_with_path[j].split('\\')[-1].split('-')[-1] + "_" +
                                                        DCM_used_path.split('\\')[-1] + ".nii.gz")

                        sitk.WriteImage(dicom_image, nifti_image_name)
                        print(f"Writing {nifti_image_name} to file was successful...\n")

    print("Done!")