import pandas as pd
import numpy as np
import SimpleITK
import os
import matplotlib.pyplot as plt

import dipy
from dipy.align.reslice import reslice

from src.utils.utils import HiddenPrints
from src.data.util.utils import get_ref_dic_from_pd, get_ref_dic_from_db
from src.db.read import read_patient_image, read_exams

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

base_path = "data/interim/train/"

target_path = "data/interim/train_registered/"   

def get_isotropic_image(info):
    # ProxID
    # DCMSerDescr
    # WorldMatrix
    # VoxelSpacing
    #
    
    exam_dir = os.path.join(base_path, info["ProxID"], info["DCMSerDescr"])
    
    if "KTrans" in info["DCMSerDescr"]:   
        mhd = [a for a in os.listdir(exam_dir) if ".mhd" in a and "T_P" not in a][0]
        path = os.path.join(exam_dir, mhd)
        itkimage = SimpleITK.ReadImage(path)
        
        image_array = SimpleITK.GetArrayFromImage(itkimage)
        image_array = np.moveaxis(image_array, 0, -1)

        affine = np.fromstring(info["WorldMatrix"], sep=",").reshape((4,4))
        zoom = itkimage.GetSpacing()
        
    else:
        reader = SimpleITK.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(exam_dir)
        reader.SetFileNames(dicom_names)
        dicom_series = reader.Execute()

        dicom_array = SimpleITK.GetArrayFromImage(dicom_series)
        image_array = np.moveaxis(dicom_array, 0, -1)

        zoom = np.fromstring(info["VoxelSpacing"], sep=",")
        affine = np.fromstring(info["WorldMatrix"], sep=",").reshape((4,4))
    
    data, affine = reslice(image_array, affine, zoom, (1., 1., 1.))
    
    return data, affine

def _center_of_mass_registration(static, static_grid2world,
                                          moving, moving_grid2world):
    
    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)
    
    return c_of_mass

def _affine_registration(static, static_grid2world,
                                          moving, moving_grid2world, c_of_mass):
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    
    level_iters = [10000, 1000, 100]
    
    sigmas = [3.0, 1.0, 0.0]
    
    factors = [4, 2, 1]
    
    affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)
    
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)
    
    return translation, affreg

def _rigid_body_registration(static, static_grid2world,
                                          moving, moving_grid2world, translation, affreg):
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)
    
    return rigid

def apply_co_registration(static, static_grid2world,
                                          moving, moving_grid2world, reference_image=None):
    
    with HiddenPrints():
        c_o_m = _center_of_mass_registration(static, static_grid2world, moving, moving_grid2world)
        affine, affreg = _affine_registration(static, static_grid2world, moving, moving_grid2world, c_o_m)
        rigid = _rigid_body_registration(static, static_grid2world, moving, moving_grid2world, affine, affreg)
    
    return rigid.transform(moving)



def register_image(moving, reference_image_type = "t2_tse_tra", overwrite = False):
    """Registers two images, using data from the database and not from the .csv file

    Parameters
    ----------
    moving: A set from of the type DB.Images. This is the target image of the registration
    reference_image_type="t2_tse_tra": The image that will serve as the standard
    overwrite=False: Is is set to true, in case the image has been already registered, it will delete it none the less and replace with the new

    Returns
    -------

    """
    moving = get_ref_dic_from_db(moving)
    
    reference = read_patient_image(moving["ProxID"], reference_image_type)

    reference = get_ref_dic_from_db(reference)

    target_patient_folder = os.path.join(target_path, moving["ProxID"])

    if not os.path.isdir(target_patient_folder):
        print("{} not yet processed. Starting now".format(patient))
        os.makedirs(target_patient_folder)
    
    # Load reference image 
    ref_image, ref_affine = get_isotropic_image(reference)
    # Check if is already exists
    img_path = os.path.join(target_patient_folder, "{}.npy".format(reference["DCMSerDescr"]))
    if not os.path.isfile(img_path):
        np.save(img_path, ref_image)



    moving_path = os.path.join(target_patient_folder, "{}.npy".format(moving["DCMSerDescr"]))

    moving_exists = os.path.isfile(img_path)
    if not moving_exists or overwrite:
        print("\t Starting image {}".format(moving["DCMSerDescr"]))

        moving_image, moving_affine = get_isotropic_image(moving)
        transformed = apply_co_registration(ref_image, ref_affine, moving_image, moving_affine )

        np.save(moving_path, transformed)


def register_exam(exam_type):
    result = read_exams(exam_type)

    for a in result:
        register_image(a, overwrite=True)

    


def register_images():
    metadata = pd.read_csv("data/interim/train_information.csv")

    exams_to_consider = ['ep2d_diff_tra_DYNDIST', 'ep2d_diff_tra_DYNDISTCALC_BVAL', 'ep2d_diff_tra_DYNDIST_ADC','t2_tse_tra', 'tfl_3d PD ref_tra_1.5x1.5_t3', 'KTrans']

    metadata = metadata[metadata.DCMSerDescr.isin(exams_to_consider)]
    patient_images = metadata[["ProxID", "DCMSerDescr", "VoxelSpacing", "WorldMatrix"]].drop_duplicates(["ProxID", "DCMSerDescr"]).groupby(['ProxID'])

    for patient, patient_dataset in patient_images:
        print('ID: ' + str(patient))
        
        try:
            reference = patient_dataset[patient_dataset.DCMSerDescr == "t2_tse_tra"].iloc[0]
        except:
            continue
        
        target_patient_folder = os.path.join(target_path, patient)

        if not os.path.isdir(target_patient_folder):
            print("{} not yet processed. Starting now".format(patient))
            os.makedirs(target_patient_folder)

        reference = get_ref_dic_from_pd(reference)
        ref_image, ref_affine = get_isotropic_image(reference)
        
        
        img_path = os.path.join(target_patient_folder, "{}.npy".format(reference["DCMSerDescr"]))
        np.save(img_path, ref_image)
        
        for b, row in patient_dataset.iterrows():
            
            
            if row.DCMSerDescr == "t2_tse_tra":
                continue
            else:
                img_path = os.path.join(target_patient_folder, "{}.npy".format(row["DCMSerDescr"]))

                if not os.path.isfile(img_path):
                    print("\t Starting image {}".format(row.DCMSerDescr))

                    row = get_ref_dic_from_pd(row)

                    moving_image, moving_affine = get_isotropic_image(row)
                    transformed = apply_co_registration(ref_image, ref_affine, moving_image, moving_affine )

                    np.save(img_path, transformed)


if __name__ == "__main__":
    register_exam("KTrans")
