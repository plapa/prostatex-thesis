import pickle
import numpy as np
import SimpleITK
import os

def load_dicom_series(input_dir):
    """Reads an entire DICOM series of slices from 'input_dir' and returns its pixel data as an array."""

    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)

    try:
        dicom_series = reader.Execute()
        dicom_array = SimpleITK.GetArrayFromImage(dicom_series)
        dicom_array = np.moveaxis(dicom_array, 0, -1)

    except:
        return None
    return dicom_array


def load_ktrans(input_dir):
    try:
        files = os.listdir(input_dir)

        mhd_file = [f for f in files if ".mhd" in f]
        mhd_file = mhd_file[0]


        path = os.path.join(input_dir, mhd_file)
        itkimage = SimpleITK.ReadImage(path)
        ct_scan = SimpleITK.GetArrayFromImage(itkimage)
        
        ct_scan = np.moveaxis(ct_scan, 0, -1)

        return ct_scan
    except: 
        return None


def get_exam(lesion_info, exam='ADC', padding=None, base_path="data/interim/train/"):
    """ Description
    :type lesion_info: row of metada_labels
    :param lesion_info:

    :type exam: string
    :param exam: the exam string to split in the description

    :type padding: int
    :param padding: Padding around the lesion to be retrieved

    :type base_path: string
    :param base_path: path to where the image files are in the project directory

    :raises:

    :rtype:
    """

    if padding is None:
        from src.helper import get_config

        config = get_config()

        padding = config["general"]["padding"]

    exam_row = lesion_info
    
    exam_row = exam_row.loc[exam_row.DCMSerDescr.str.contains(exam)]
    
    if exam_row.empty:
        print("No lesion found")
        return None
    
    else:
        tmp_row = exam_row.iloc[0]
        
        exam_folder = os.path.join(os.getcwd(), base_path, tmp_row.ProxID, tmp_row.DCMSerDescr)
    
        if(exam != 'KTrans'):
            image = load_dicom_series(input_dir=exam_folder)

        else:
            image = load_ktrans(exam_folder)

        if image is None:
            return None
        
        if(tmp_row.k < image.shape[2]):
            slice_array = image[:,:, tmp_row.k]
            patch = get_image_patch(slice_array, (tmp_row.i, tmp_row.j), padding=padding)
            # print(patch.shape)

        else:
            print("Had to cut image")
            return None
            
        if(patch.shape == (2*padding,2*padding)):
            return np.asarray(patch).reshape((1,2*padding,2*padding, 1))