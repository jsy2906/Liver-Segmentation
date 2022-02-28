from glob import glob
import shutil
import os

import dicom2nifti

import nibabbel as nib
import numpy as np


# Create groups of element_num slices
def create_group(in_path, out_path, element_num):
    '''
    <Create groups of element_num slices>
    in_path : dicom file path
    out_path : grout dicom file path
    element_num : group으로 나눌 수

    dicome file을 지정해준 element_num만큼 slice로 나눈 후,
    그룹으로 만들어 지정된 output 경로에 저장해줌
    '''
    for patient in glob(in_path + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))
        num_folders = int(len(glob(patient + '/*')) / element_num)

        for i in range(num_folders):
            output_path_name = os.path.join(out_path, patient_name+'_'+str(i))
            os.mkdir(output_path_name)
            for i, file in enumerate(glob(patient + '/*')):
                if i == element_num:
                    break
                shutil.move(file, output_path_name)


# Convert the dicom group files into nifti files
def convert_nifti(in_path_dicom_images, in_path_dicom_labels, out_path_images, out_path_labels):
    '''
    <Convert the dicom group files into nifti files>
    in_path_dicom_images : dicom group image path
    in_path_dicom_labels : dicom group label path
    out_path_images : image nifti file path
    out_path_labels : label nifti file path

    그룹으로 묶어준 dicom 파일을 nifti 파일로 변경
    '''
    list_images = glob(in_path_dicom_images + '/*')
    list_labels = glob(in_path_dicom_labels + '/*')
    
    # Change images
    if not os.path.isdir(out_path_images):
        os.makedirs(out_path_images)
    for image_patient in list_images:
        image_patient_name = os.path.basename(os.path.normpath(image_patient))
        dicom2nifti.dicom_series_to_nifti(image_patient, os.path.join(out_path_images, image_patient_name+'.nii.gz'))

    # Change labels
    if not os.path.isdir(out_path_labels):
        os.makedirs(out_path_labels)
    for label_patient in list_labels:
        label_patient_name = os.path.basename(os.path.normpath(label_patient))
        dicom2nifti.dicom_series_to_nifti(label_patient, os.path.join(out_path_labels, label_patient_name+'.nii.gz'))

    
    # Find not empty label files
    def find_empty(input_nifti_image, input_nifti_label, out_path_images, out_path_labels):
        '''
        <Find not empty label files>
        input_nifti_image : image nifti file path
        input_nifti_label : label nifti file path
        out_path_images : not empty image nifti file path
        out_path_labels : not empty label nifti file path

        label nifti file 중 비어있지 않은 label 파일을 찾아 image와 label을 지정 output 경로에 복사
        '''
        list_labels = glob(input_nifti_label + '/*')

        if not os.path.isdir(out_path_images):
            os.mkdir(out_path_images)

        if not os.path.isdir(out_path_labels):
            os.mkdir(out_path_labels)

        for patient in list_labels:
            nifti_file = nib.load(patient)
            fdata = nifti_file.get_fdata()
            np_unique = np.unique(fdata)
            if len(np_unique) > 1:
                shutil.copy(patient, out_path_labels)

                patient_name = os.path.basename(os.path.normpath(patient))
                shutil.copy(input_nifti_image+patient_name, out_path_images)

