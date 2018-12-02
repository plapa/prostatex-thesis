import pandas as pd 


import src.db.base
from src.db.tables import *
from src.db.main import s, engine
from sqlalchemy import select, func, literal_column

def lesion_image_coordinates(filter_exams = ["t2_tse_tra","ep2d_diff_tra_DYNDIST", "ep2d_diff_tra_DYNDISTCALC_BVAL"]):
    ''' This query returns  lesion coodinates for each image

    EXAMPLE:
            ProxID  fid                       imagetype  reg_i  reg_j  reg_k
    0     ProstateX-0000    1                          KTrans     83    112     27

    '''
    q = s.query(Patient.ProxID, Lesion.fid, Image.imagetype, Lesion.reg_i, Lesion.reg_j, Lesion.reg_k, Lesion.clin_sig ).join(Image).join(Lesion).filter(Image.registered==1)

    if filter_exams is not None:
        q = q.filter(Image.imagetype.in_(filter_exams))


    # This query retrieves every patient that has all the images required. So that we don't have patients with only 1 or 2 of the required 3 exames, e.g.
    every_image = s.query(Patient.ProxID, (func.count(Patient.ProxID)).label('counts')).join(Image).filter(Image.registered==1)
    if filter_exams is not None:
        every_image = every_image.filter(Image.imagetype.in_(filter_exams))
    every_image = every_image.group_by(Patient.ProxID).having(func.count(Patient.ProxID) == len(filter_exams))
    every_image = every_image.subquery()

    #print(a.all())

    q = q.join(every_image).filter(Patient.ProxID == every_image.c.ProxID)

    result = pd.read_sql(q.statement, engine)

    return result

def lesion_significance(proxid, coords):

    q = s.query(Lesion.patient_id, Lesion.fid, Lesion.reg_i, Lesion.reg_j, Lesion.reg_k, Lesion.clin_sig).filter(Lesion.patient_id==proxid, Lesion.reg_i == coords[0], Lesion.reg_j == coords[1], Lesion.reg_k==coords[2])
    result = pd.read_sql(q.statement, engine)

    return result.iloc[0]



if __name__ == "__main__":
    print(lesion_significance("ProstateX-0000", (83, 112, 27)))
