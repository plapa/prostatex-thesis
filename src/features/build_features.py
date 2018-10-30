from src.helper import get_config
from imgaug import augmenters as iaa
import numpy as np
from transformations import normalize_01, normalize_meanstd, create_augmenter

def apply_transformations(X = None, y = None, save = False):

    config = get_config()

    # Load X, y

    if X is None:
        X = np.load("data/processed/X_2c.npy")
        y = np.load("data/processed/y_2c.npy")

    if config["preprocessing"]["rescale"]:
        
        if config["preprocessing"]["rescale_method"] == "normalize":
            X_ = normalize_01(X)
        
        elif config["preprocessing"]["rescale_method"] == "standartize":
            X_ = normalize_01(X)

        else: 
            print("{} is not a valid rescaling method".format(config["preprocessing"]["rescale_method"]))
    
    else:
        X_ = X

    aug = create_augmenter()

    samples_to_augment = config["preprocessing"]["augmented_ds_size"]

    X_augmented = np.empty((samples_to_augment, 64, 64, 3))
    y_ = []


    i = 0
    for index, item in enumerate(X_augmented):
        rnd = np.random.choice(X.shape[0], 1)[0]
        
        
        X_augmented[index, :, :, : ] = X[rnd]
        y_.append(y[rnd])

    X_aug = aug.augment_images(X_augmented)


    return X_aug, y_

if __name__ == "__main__":
    X, y = apply_transformations()

    print(X.shape)