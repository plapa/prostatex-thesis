import sys
from os import listdir, walk
import os
import shutil



base_path = "../../data/raw/train_mhd"

destination_path = "../../data/interim/train_mhd"

patient_folders = os.listdir(base_path)


for patient in patient_folders:
    exam_folder = os.path.join(base_path,patient)

    exam_files = os.listdir(exam_folder)

    # Check if patient_folder already exists:

    d_patient_folder = os.path.join(destination_path, patient, "KTrans")

    if not os.path.isdir(d_patient_folder):
        os.makedirs(d_patient_folder)

    for exam in exam_files:

        curr_file_path = os.path.join(base_path, patient, exam)

        dest_file_path = os.path.join(d_patient_folder, exam)
        
        
        shutil.copyfile(curr_file_path, dest_file_path)


    

