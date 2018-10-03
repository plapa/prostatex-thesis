import pickle
import numpy as np
import SimpleITK
import os


def get_image_patch(img_array, coords, padding=32):


    """ Description
    :type img_array: numpy array
    :param img_array: a image comprised of a numpy array

    :type coords: list(i,j)
    :param coords: The coordinates where the crop will be centered, of type (i,j), i being the columns and j the row

    :type padding: int or list(x,y)
    :param padding: The padding that will be around the center coordinates. If an int, it will create a square image. If a list x is the horizontal padding and y the vertical

    :raises:

    :rtype:
     """

    i = coords[0]
    j = coords[1]

    if isinstance(padding, list):
        h_padding = padding[0]
        v_padding = padding[1]
    else:
        h_padding = padding
        v_padding = padding

    X_ = img_array[j - v_padding: j + v_padding, i - h_padding: i + h_padding]
    # NUMPY ARRAYs are of standart (row, columns)

    return X_



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


def get_exam(lesion_info, exam='ADC', padding=32, base_path="../data/interim/train/"):
    
    
    exame_row = lesion_info
    
    exame_row = exame_row.loc[exame_row.DCMSerDescr.str.contains(exam)]
    
    if exame_row.empty:
        print("No lesion found")
        return None
    
    else:
        tmp_row = exame_row.iloc[0]
        
        exam_folder = os.path.join(base_path, tmp_row.ProxID, tmp_row.DCMSerDescr)
    
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