import pandas as pd
import numpy as np
import os
from src.db.main import engine
from src.db.base import Base
from src.db.tables import *

metadata = pd.read_csv("data/interim/train_information.csv")
base_path = "data/interim/train_registered/"

def load_patients():
    patients = pd.DataFrame(metadata.ProxID.unique())
    patients.columns = ["ProxID"]
    patients.head()

    patients.to_sql("patients", con=engine, if_exists="append", index=False)

def load_lesions():
    def get_coordinates(ijk, spacing):
        a = np.fromstring(str(ijk), sep=" ")
        b = np.fromstring(str(spacing), sep= ",")[:3]
        coords = a*b
        return int(coords[0]), int(coords[1]), int(coords[2])
    
    lesion_coordinates = metadata[metadata.DCMSerDescr == "t2_tse_tra"].drop_duplicates(["ProxID", "ijk"])[["ProxID", "ijk", "VoxelSpacing","zone", "ClinSig", "fid"]]    
    g = lambda x: pd.Series(get_coordinates(x.ijk, x.VoxelSpacing), dtype=int)
    lesion_coordinates[["reg_i", "reg_j", "reg_k"]] = lesion_coordinates[["ijk", "VoxelSpacing"]].apply(g,  axis=1)
    
    lesion_coordinates = lesion_coordinates[["ProxID", "fid", "zone", "reg_i", "reg_j", "reg_k", "ClinSig"]]
    lesion_coordinates.rename(columns={"ClinSig" : "clin_sig", "ProxID": "patient_id"}, inplace=True)

    # To make sure that dfferne coordinates mean different lesions
    lesion_coordinates["ijk"] = lesion_coordinates[["reg_i", "reg_j", "reg_k"]].apply(lambda x: ''.join(str(x.values)), axis=1)
    
    lesion_coordinates.to_sql('lesions', con=engine, if_exists="append", index=False)

def load_images():
    
    def check_reg_img_exists(pat, exam):
        file_name = '{}.npy'.format(exam)
        image_path = os.path.join(base_path, pat, file_name)

        return os.path.isfile(image_path)
    
    
    images = metadata[["ProxID", "DCMSerDescr", "VoxelSpacing", "WorldMatrix","imagetype"]].drop_duplicates(["ProxID", "DCMSerDescr"])

    g = lambda x: pd.Series(check_reg_img_exists(x.ProxID, x.DCMSerDescr))

    images[["registered"]] = images[["ProxID", "DCMSerDescr"]].apply(g, axis=1)
    
    images.rename(columns={"ProxID" : "patient_id", "DCMSerDescr": "dcmser_descr", "VoxelSpacing": "voxel_spacing", "WorldMatrix": "world_matrix", "imagetype": "image_type"}, inplace=True)
    
    images.to_sql("images", con=engine, if_exists="append", index=False)

def update_image(image):
   stmt = User.update().\
       values(no_of_logins=(User.no_of_logins + 1)).\
       where(User.username == form.username.data)
   conn.execute(stmt)

   stmt = Image.update().values(registered = True.where(Image.image_id == image["image_id"]))


def load_all():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    load_patients()
    load_images()
    load_lesions()

if __name__=="__main__":
    load_all()