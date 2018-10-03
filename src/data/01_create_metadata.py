import os
import pandas as pd

images_info_path = "../data/raw/Train_Information/ProstateX-Images-Train.csv"
findings_info_path = "../data/raw/Train_Information/ProstateX-Findings-Train.csv"
mhd_images_info_path= "../data/raw/Train_Information/ProstateX-Images-KTrans-Train.csv"


images_info = pd.read_csv(images_info_path)
findings_info = pd.read_csv(findings_info_path)

mhd_image_info = pd.read_csv(mhd_images_info_path)
mhd_image_info["DCMSerDescr"] = "KTrans"


mri_metadata = pd.merge(images_info, findings_info, left_on=["ProxID", "fid", "pos"], right_on=["ProxID", "fid", "pos"])
mhd_metadata = pd.merge(mhd_image_info, findings_info, left_on=["ProxID", "fid", "pos"], right_on=["ProxID", "fid", "pos"])

metadata_labels = pd.concat([mri_metadata, mhd_metadata], sort= False)
metadata_labels[['i', 'j', 'k']] = metadata_labels["ijk"].str.split(" ", expand=True)

metadata_labels.to_csv("../data/interim/train_information.csv")