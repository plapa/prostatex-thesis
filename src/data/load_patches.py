import pandas as pd 
import numpy as np
import warnings
import os

from src.db.read import lesion_image_coordinates,  lesion_significance
from src.data.util.utils import get_image_patch
from src.helper import get_config

base_path = "data/interim/train_registered/"

config = get_config()

base_path = config["meta"]["registered_path"]



def create_dataset(overwrite=False, note = ""):

    # Just loading some of the general configurations
    padding = config["general"]["padding"]
    to_consider = config["meta"]["exams_to_consider"]

    # Some patients have the same exam id for different lesions, so we need to differentiate them throught their coordinates
    # Also some patients have the same lesion more than once, so we need to only choose one
    lesion_info = lesion_image_coordinates(to_consider)
    lesion_info["ijk"] = lesion_info[["reg_i", "reg_j", "reg_k"]].apply(lambda x: ''.join(str(x.values)), axis=1)
    lesion_info = lesion_info.drop_duplicates(["ProxID", "imagetype", "ijk"])

    # Just creating some variable to store descriptive information
    n_existing_patients = lesion_info.drop_duplicates(["ProxID"]).shape[0]
    n_existing_lesions = lesion_info.drop_duplicates(["ProxID", "ijk"]).shape[0]
    checked_patients = []

    # For each lesion group every image
    engroup = lesion_info.groupby(["ProxID", "ijk", "clin_sig"])

    i = 0
    X = np.empty((1000, 2*padding,2*padding,3))
    y = []

    # We are going to iterate through every lesion, and retrieve the corresponding patch for every image.async for every image
    # They are going to be store in a array, a channel for each image.

    for patient_lesion, subdf in engroup:

        patient_folder = os.path.join(base_path, patient_lesion[0])
        lesion_image = np.empty(shape=(1,2*padding,2*padding,0))

        for info, image in subdf.iterrows():
            file_name = "{}.npy".format(image.imagetype)
            image_path = os.path.join(patient_folder, file_name)
            image_file = np.load(image_path)

            coords = (image.reg_i, image.reg_j, image.reg_k)
            file_patch = get_image_patch(image_file, coords , padding)

            # if the patch is none, then not a valid image of the lesion could have ben retrieved
            # then the current lesion is discarded
            if file_patch is None:
                lesion_image = None
                warnings.warn("A lesion could not have been retrieved from the image.")
                break
            else:
                lesion_image = np.concatenate([lesion_image, file_patch], axis = 3)


        
        if lesion_image is not None:
            X[i, :, :, :] = lesion_image
            y.append(patient_lesion[2])
            checked_patients.append(patient_lesion[0])
            i = i+1

    X = X[:i]
    y = np.array(y)


    print(" ### PATCHES FINISHED CREATING ### ")
    print("Number of existing patients: {}".format(n_existing_patients))
    print("Number of patients used: {}".format(np.unique(np.array(checked_patients)).shape[0]))

    print("Number of existing lesions : {}".format(n_existing_lesions))
    print("Number of lesions used: {}".format(i))

    print("X SHAPE: {}".format(X.shape))
    print("y SHAPE: {}".format(y.shape))

    if overwrite:
        print("SAVING AND OVERWRITING EXISTING FILES")
        np.save("data/processed/X_{}.npy".format(note), X)
        np.save("data/processed/y_{}.npy".format(note), y)


if __name__ == "__main__":
    create_dataset(True, "tf_t2_kt")