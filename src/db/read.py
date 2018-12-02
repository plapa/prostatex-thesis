import pandas as pd 


import src.db.base
from src.db.tables import *
from main import s, engine

def lesion_image_coordinates():
    ''' This query returns  lesion coodinates for each image

    EXAMPLE:
            ProxID  fid                       imagetype  reg_i  reg_j  reg_k
    0     ProstateX-0000    1                          KTrans     83    112     27

    '''
    q = s.query(Patient.ProxID, Lesion.fid, Image.imagetype, Lesion.reg_i, Lesion.reg_j, Lesion.reg_k ).join(Image).join(Lesion).filter(Image.registered==1)

    result = pd.read_sql(q.statement, engine)

    return result

if __name__ == "__main__":
    print(lesion_image_coordinates())