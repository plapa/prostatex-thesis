import pickle
import numpy as np

def create_mask(shape, obs, k):
    mask_ = np.zeros(shape, dtype=int, order='C')

    relevant_coordinates = obs[(obs.k == k)]
    for rel_row in relevant_coordinates.itertuples():
        mask_[rel_row.i, rel_row.j] = 2 if rel_row.ClinSig else 1
        
    return mask_

def save_pickle(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)