import pandas as pd
import numpy as np
import SimpleITK
import os
from imgaug import augmenters as iaa


from src.data.util.utils import get_image_patch, load_dicom_series, get_exam, load_ktrans
from src.helper import get_config

def create_dataset(padding=None, overwrite=False):

    config = get_config()
    base_path = "data/interim/train/"

    metadata = pd.read_csv("data/interim/train_information.csv")

    s = 0
    i = 0

    if padding is None:

        padding = config["general"]["padding"]

        print(padding)

    X = np.empty((1000, 2*padding,2*padding,3))

    y = []

    i = 0
    to_iterate = metadata.drop_duplicates(["ProxID", "fid", "pos"])[["ProxID", "fid", "ClinSig"]]

    aug = iaa.Scale(0.5, interpolation=config["preprocessing"]["interpolation"])

    for tup in to_iterate.itertuples():
        X_ = []
        
        
        lesion_info = metadata[(metadata.ProxID == tup.ProxID) & (metadata.fid == tup.fid)]
        

        adc = get_exam(lesion_info, '_ADC', padding=padding )
        if adc is  None:
            continue
            

        t2_tse = get_exam(lesion_info, 't2_tse_tra', padding=2*padding)
        if t2_tse is None:
            continue
        t2_tse = np.uint8(t2_tse)
        t2_tse = aug.augment_images(t2_tse)
        
        ktrans = get_exam(lesion_info, 'KTrans', padding=padding)
        
        if ktrans is None:
            continue

        X_ = np.concatenate([adc, t2_tse, ktrans], axis = 3)

        X[i, :, :, :] = X_
        i = i+1
        
        y_ = 1 if tup.ClinSig else 0
        y.append(y_)
        
        

    X = X[:i]

    print(X.shape)



    if overwrite:
        np.save("data/processed/X_36.npy", X)
        np.save("data/processed/y_36.npy", y)

    return X, y

if __name__ == "__main__":
    x,y = create_dataset(overwrite=True)
    print(x.shape)