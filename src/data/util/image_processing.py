import pandas as pd
import numpy as np
import SimpleITK
import dipy
import os
from dipy.align.reslice import reslice

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

from src.helper import get_config

from src.utils.utils import HiddenPrints

config = get_config()

target_path = config["meta"]["registered_path"]
base_path = config["meta"]["unregistered_path"]

  

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