import os
import pandas as pd


def create_metadata(return_metadata = False):
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
    #metadata_labels[['i', 'j', 'k']] = metadata_labels["ijk"].str.split(" ", expand=True)

    # Save the data frame onto disk
    metadata_labels.to_csv("data/interim/train_information.csv")

    if return_metadata:
        return metadata

if __name__ == "__main__":
    create_metadata()