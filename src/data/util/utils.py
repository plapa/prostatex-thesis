import pickle
import numpy as np
import SimpleITK
import os


def get_image_patch(img_array, coords, padding=None):
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

    if padding is None:
        from src.helper import get_config

        config = get_config()

        padding = config["general"]["padding"]

    i = coords[0]
    j = coords[1]
    k = coords[2]
    


    if isinstance(padding, list):
        h_padding = padding[0]
        v_padding = padding[1]
    else:
        h_padding = padding
        v_padding = padding
        
    try:
        X_ = img_array[j - v_padding : j + v_padding, i - h_padding : i + h_padding, k]
    except:
        print(img_array.shape)
        print(coords)
        return None
    else:    
        if(X_.shape == (2*padding,2*padding)):
            return np.asarray(X_).reshape((1,2*padding,2*padding, 1))



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

def get_ref_dic_from_pd(reference):

    reference = {"ProxID":reference.ProxID, "DCMSerDescr":reference.DCMSerDescr,
    "WorldMatrix":reference.WorldMatrix, "VoxelSpacing":reference.VoxelSpacing }
    
    return reference

def get_ref_dic_from_db(reference):

    reference = {"ProxID":reference.patient_id, "DCMSerDescr":reference.imagetype,
    "WorldMatrix":reference.world_matrix, "VoxelSpacing":reference.voxel_spacing }
    
    return reference
