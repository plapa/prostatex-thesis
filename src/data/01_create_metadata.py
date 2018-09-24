import os
import pandas as pd

images_info_path = "../data/raw/Train_Information/ProstateX-Images-Train.csv"
findings_info_path = "../data/raw/Train_Information/ProstateX-Findings-Train.csv

images_info = pd.read_csv(images_info_path)

findings_info = pd.read_csv(findings_info_path)

metadata_labels = pd.merge(images_info, findings_info, left_on=["ProxID", "fid", "pos"], right_on=["ProxID", "fid", "pos"])
metadata_labels[['i', 'j', 'k']] = metadata_labels["ijk"].str.split(" ", expand=True)

metadata_labels.to_csv("../data/interim/train_information.csv")
