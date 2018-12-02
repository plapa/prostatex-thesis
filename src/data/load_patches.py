import pandas as pd 
import numpy as np
import os

from src.db.read import lesion_image_coordinates,  lesion_significance
from src.data.util.utils import get_image_patch


base_path = "data/interim/train_registered/"


def main():
    to_consider = ["tfl_3d PD ref_tra_1.5x1.5_t3", "t2_tse_tra", "ep2d_diff_tra_DYNDIST_ADC"]

    lesion_info = lesion_image_coordinates(to_consider)
    lesion_info["ijk"] = lesion_info[["reg_i", "reg_j", "reg_k"]].apply(lambda x: ''.join(str(x.values)), axis=1)
    lesion_info = lesion_info.drop_duplicates(["ProxID", "imagetype", "ijk"])

    engroup = lesion_info.groupby(["ProxID", "ijk", "clin_sig"])

    i = 0
    X = np.empty((1000, 64,64,3))
    y = []

    for patient_lesion, subdf in engroup:
        patient_folder = os.path.join(base_path, patient_lesion[0])
        patient_image = np.empty(shape=(1,64,64,0))
        for info, image in subdf.iterrows():
            file_name = "{}.npy".format(image.imagetype)
            image_path = os.path.join(patient_folder, file_name)
            image_file = np.load(image_path)

            coords = (image.reg_i, image.reg_j, image.reg_k)

            file_patch = get_image_patch(image_file, coords , 32)
            if file_patch is None:
                patient_image = None
                break
            else:
                patient_image = np.concatenate([patient_image, file_patch], axis = 3)


        
        if patient_image is not None:
            X[i, :, :, :] = patient_image
            y.append(patient_lesion[2])
            i = i+1

    X = X[:i]
    y = np.array(y)

    print(X.shape)
    print(y.shape)


if __name__ == "__main__":
    main()