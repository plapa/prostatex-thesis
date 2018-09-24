import pandas as pd
import numpy as np
import pydicom
from util.Observation import Observation
from util.utils import create_mask, save_pickle
import pickle
import os

base_path = "../data/interim/train/"

metadata = pd.read_csv("../data/interim/train_information.csv")
metadata = metadata.drop_duplicates(subset = ["ProxID", "DCMSerDescr"])
metadata = metadata.apply(pd.to_numeric,errors='ignore')


observations = []
for row in metadata.itertuples():

    image_path = os.path.join(base_path, row.ProxID, row.DCMSerDescr)
    image_list = os.listdir(image_path)
    image_list.sort()

    X_ = None
    y_ = None


    try:
        image_list = os.listdir(image_path)
    except FileNotFoundError:
        continue
    image_list.sort()

    y_meta_data = metadata[(metadata.ProxID == row.ProxID) & (metadata.DCMSerDescr == row.DCMSerDescr)]


    tmp_path = os.path.join(image_path, image_list[0])



    for img in image_list:       
        img_path = os.path.join(image_path, img)
        img_array = pydicom.read_file(img_path).pixel_array
        
        k = int(img.split(".")[0])
        
        relevant_coordinates = y_meta_data[(y_meta_data.k == k)]
        
        if X_ is None:
            X_ = img_array
            y_ = create_mask(img_array.shape, relevant_coordinates, k)
        else:
            
            mask_ = create_mask(img_array.shape, relevant_coordinates, k)
            
            
            X_ = np.dstack([X_, img_array])
            y_ = np.dstack([y_, mask_])
            
    temp_obs = Observation()
    temp_obs.X_ = X_
    temp_obs.y_ = y_
    temp_obs.description = row.DCMSerDescr
    temp_obs.ProxID = row.ProxID


    observations.append(temp_obs)
    print("Observation imported")

pickle_path = "../data/interim/train.pkl"

save_pickle(observations, pickle_path)