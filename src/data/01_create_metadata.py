import os
import pandas as pd

from src.data.util.utils import replace_wrapper

def create_metadata(return_metadata = False, save=False):
    # This file merges the information from all the 3 train files and creates a new one to be used to identify the lesions
    images_info_path = "data/raw/Train_Information/ProstateX-Images-Train.csv" # File that contains MRI information
    findings_info_path = "data/raw/Train_Information/ProstateX-Findings-Train.csv" # File that contains lesion information (target)
    mhd_images_info_path= "data/raw/Train_Information/ProstateX-Images-KTrans-Train.csv" # File that contains KTRans information

    # Loads the information about the MRI exames and the lesion info
    images_info = pd.read_csv(images_info_path)
    findings_info = pd.read_csv(findings_info_path)

    # Load the KTrans images and add a new columns that will be used to pass the exam files
    mhd_image_info = pd.read_csv(mhd_images_info_path)
    mhd_image_info["DCMSerDescr"] = "KTrans"

    # Merges both dataframes of the MRI and KTRANS with the lesion information
    mri_metadata = pd.merge(images_info, findings_info, left_on=["ProxID", "fid", "pos"], right_on=["ProxID", "fid", "pos"])
    mhd_metadata = pd.merge(mhd_image_info, findings_info, left_on=["ProxID", "fid", "pos"], right_on=["ProxID", "fid", "pos"])

    # Concatenates both data frames and creates a new column that contains the lesion coordinates
    metadata_labels = pd.concat([mri_metadata, mhd_metadata], sort= False)

    metadata_labels["imagetype"] = metadata_labels["DCMSerDescr"]


    metadata_labels["imagetype"] = replace_wrapper(metadata_labels["imagetype"], "BVAL", "BVAL_TRA")
    metadata_labels["imagetype"] = replace_wrapper(metadata_labels["imagetype"], "ADC", "ADC_TRA")
    metadata_labels["imagetype"] = replace_wrapper(metadata_labels["imagetype"], "LOC", "LOC_TRA")
    metadata_labels["imagetype"] = replace_wrapper(metadata_labels["imagetype"], "t2_tse_tra", "T2_TSE_TRA")
    metadata_labels["imagetype"] = replace_wrapper(metadata_labels["imagetype"], "PD", "PD_TRA")
    metadata_labels["imagetype"] = replace_wrapper(metadata_labels["imagetype"], "ep2d_diff_tra", "EP2D_DIFF_TRA")


    # Save the data frame onto disk

    if save:
        metadata_labels.to_csv("data/interim/train_information.csv")

    if return_metadata:
        return metadata_labels

if __name__ == "__main__":
    create_metadata(save=True)