import sys
import pydicom
from os import listdir, walk
import os
import shutil


def rename_files(source_path = None, destination_path = None):

    if source_path is None:     
        source_path = "data/raw/Test_Images"

    if destination_path is None:
        destination_path = "data/interim/test"

    all_substrc = [f for f in walk(source_path)]
    f = []

    '''
    This scripts alters the original structured of the files, from a very messy folder naming and hierarchy 
    to a more organized and succint one.

    This aims to make it easier to merge the images with the target labels.
    '''
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source_path) for f in filenames]

    for file in result:
        #print(file)

        try:
            temp_dicom = pydicom.read_file(file)
        except:
            print("DICOM ERROR. SKIPPING. PATH: {}".format(file))
            continue



        # print("{}/-{}/{}-".format(temp_dicom.PatientName, temp_dicom.SeriesDescription, 1))


        # Check if patient folder exists

        patient_folder = os.path.join(destination_path, str(temp_dicom.PatientName))
        exam_folder = os.path.join(patient_folder, str(temp_dicom.SeriesDescription)) 

        if not os.path.isdir(patient_folder):
            os.makedirs(patient_folder)

        if not os.path.isdir(exam_folder):
            os.makedirs(exam_folder)

        file_len = len(file)

        # file name example: "000001.dcm"

        file_name = file[file_len - 10:]
        dest_path = os.path.join(exam_folder, file_name)
        # print(file_name)


        shutil.copyfile(file, dest_path)

    print("Operation completed")

def reorder_slices(base_path=None):



    if base_path is None:
        base_path = 'data/interim/train/'
    patients = os.listdir(base_path)

    i = 0
    j = len(patients)

    for pats in patients:
        i = i + 1
        print("{} / {}".format(i, j))
        patient_path = os.path.join(base_path, pats)
        exam_list = os.listdir(patient_path)
    
        for exam in exam_list:
            exam_path = os.path.join(patient_path, exam)

            for p in os.listdir(exam_path):
                original = os.path.join(exam_path, p)
                dest = os.path.join(exam_path, "T_" + p)
                os.rename(original,dest)
            
            reorder_files(exam_path)

def reorder_files(path):
    for p in os.listdir(path):
        file_path = os.path.join(path, p)

        try:
            img_file = pydicom.read_file(file_path, force=True)
            file_name = "{}.dcm".format(str(int(img_file.InstanceNumber) - 1).rjust(6, "0"))

            dest = os.path.join(path, file_name)

            os.rename(file_path,dest)
        except:
            pass

if __name__ == "__main__":
    reorder_slices()
