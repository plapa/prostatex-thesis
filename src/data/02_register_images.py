import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.utils import HiddenPrints
from src.data.util.utils import get_ref_dic_from_pd, get_ref_dic_from_db
from src.db.read import read_patient_image, read_exams
from src.db.write import update_image

from src.data.util.image_processing import get_isotropic_image, _center_of_mass_registration, _affine_registration,_rigid_body_registration, apply_co_registration
from src.helper import get_config

config = get_config()

target_path = config["meta"]["registered_path"]

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

    reference = read_patient_image(moving["ProxID"], reference_image_type, is_registered=True)


    if reference is None:
        return
    reference = get_ref_dic_from_db(reference)

    target_patient_folder = os.path.join(target_path, moving["ProxID"])

    if not os.path.isdir(target_patient_folder):
        print("{} not yet processed. Starting now".format(moving["ProxID"]))
        os.makedirs(target_patient_folder)
    
    # Load reference image 
    ref_image, ref_affine = get_isotropic_image(reference)
    # Check if is already exists
    img_path = os.path.join(target_patient_folder, "{}.npy".format(reference["DCMSerDescr"]))
    if not os.path.isfile(img_path):
        np.save(img_path, ref_image)

    moving_path = os.path.join(target_patient_folder, "{}.npy".format(moving["DCMSerDescr"]))

    moving_exists = os.path.isfile(moving_path)
    if not moving_exists or overwrite:
        print("{} \t Starting image {}".format(moving["ProxID"], moving["DCMSerDescr"]))

        moving_image, moving_affine = get_isotropic_image(moving)
        transformed = apply_co_registration(ref_image, ref_affine, moving_image, moving_affine )

        np.save(moving_path, transformed)

        #update_image(moving)
    else:
        Print("Already exists")





    

#DEPRECATED
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


def register_exam(exam_type):
    result = read_patient_image(image=exam_type, enforce_one=False, is_registered=False)
    print(len(shape))

    for a in tqdm(result):
        register_image(a, overwrite=True)
        

if __name__ == "__main__":
    register_exam("LOC_tra")
