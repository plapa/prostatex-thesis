import pandas as pd
import numpy as np
import SimpleITK
import os

from util.utils import get_image_patch, load_dicom_series, get_exam, load_ktrans

def create_dataset(padding=32, overwrite=False):
    base_path = "../data/interim/train/"

    metadata = pd.read_csv("../data/interim/train_information.csv")

    s = 0
    i = 0

    padding = 32

    X = np.empty((1000, 64,64,3))

    y = []

    i = 0
    to_iterate = metadata.drop_duplicates(["ProxID", "fid", "pos"])[["ProxID", "fid", "ClinSig"]]

    for tup in to_iterate.itertuples():
        X_ = []
        
        person_path = os.path.join(base_path, tup.ProxID)       
        
        lesion_info = metadata[(metadata.ProxID == tup.ProxID) & (metadata.fid == tup.fid)]
        

        adc = get_exam(lesion_info, '_ADC', padding=padding )
        if adc is  None:
            continue
            

        t2_tse = get_exam(lesion_info, 't2_tse_tra', padding=padding)
        if t2_tse is None:
            continue
        
        ktrans = get_exam(lesion_info, 'KTrans', padding=padding)
        
        if ktrans is None:
            continue

        X_ = np.concatenate([adc, t2_tse, ktrans], axis = 3)
        # X = np.append(X, X_, axis=0)
        X[i, :, :, :] = X_
        i = i+1
        
        y_ = 1 if tup.ClinSig else 0
        y.append(y_)
        
        

    X = X[:i]

    print(X.shape)



    if overwrite:
        np.save("../data/processed/X.npy", X)
        np.save("../data/processed/y.npy", y)

    return X, y

if __name__ == "__main__":
    x,y = create_dataset()
    print(x.shape)