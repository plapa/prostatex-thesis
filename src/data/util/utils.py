import numpy as np
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
        return None
    else:    
        if(X_.shape == (2*padding,2*padding)):
            return np.asarray(X_).reshape((1,2*padding,2*padding, 1))





def get_ref_dic_from_pd(reference):

    reference = {"ProxID":reference.ProxID, "DCMSerDescr":reference.DCMSerDescr,
    "WorldMatrix":reference.WorldMatrix, "VoxelSpacing":reference.VoxelSpacing }
    
    return reference

def get_ref_dic_from_db(reference):

    reference = {"ProxID":reference.patient_id, "DCMSerDescr":reference.imagetype,
    "WorldMatrix":reference.world_matrix, "VoxelSpacing":reference.voxel_spacing }
    
    return reference
