from src.helper import get_config
from imgaug import augmenters as iaa
import numpy as np
from src.features.transformations import normalize_01, normalize_meanstd, create_augmenter


def apply_rescale(X):
    config = get_config()
    if config["preprocessing"]["rescale"]:
        
        if config["preprocessing"]["rescale_method"] == "normalize":
            X_ = normalize_meanstd(X)
        
        elif config["preprocessing"]["rescale_method"] == "standartize":
            X_ = normalize_01(X)

        else: 
            print("{} is not a valid rescaling method".format(config["preprocessing"]["rescale_method"]))
    
    else:
        X_ = X

    return X_

def apply_transformations(X = None, y = None, save = False):

    config = get_config()

    if X is None:
        X = np.load("data/processed/X_2c.npy")
        y = np.load("data/processed/y_2c.npy")


    # X_ = apply_rescale(X)

    aug = create_augmenter()

    samples_to_augment = config["preprocessing"]["augmented_ds_size"]

    X_augmented = np.empty((samples_to_augment, 64, 64, 3))
    y_ = []


    i = 0
    for index, item in enumerate(X_augmented):
        rnd = np.random.choice(X.shape[0], 1)[0]
        
        
        X_augmented[index, :, :, : ] = X[rnd]
        y_.append(y[rnd])

    X_augmented= np.concatenate([X_augmented, X], axis = 0)
    y_ = np.concatenate([y_, y])    

    X_aug = aug.augment_images(X_augmented)

    if save:
        np.save("data/processed/X_a.npy", X_augmented)
        np.save("data/processed/y_a.npy", y_)
    return X_aug, y_

def create_train_val_set(X = None, y = None):
    config = get_config()

    if X is None:
        X = np.load("data/processed/X_2c.npy")
        y = np.load("data/processed/y_2c.npy")

    train_samples = round(X.shape[0] * config["train"]["train_val_split"])

    X_train = X[:train_samples,: ,:,:]
    X_val = X[train_samples:, :,::]

    y_train = y[:train_samples]
    y_val = y[train_samples:]


    return X_train, X_val, y_train, y_val

def create_augmented_dataset(save=False, return_data = False):
    config = get_config()
    
    X_train, X_val, y_train, y_val = create_train_val_set()


    X_train, y_train = apply_transformations(X_train, y_train)

    X_val = apply_rescale(X_val)

    if save:
        np.save("data/processed/X_train.npy", X_train)
        np.save("data/processed/y_train.npy", y_train)

        np.save("data/processed/X_val.npy", X_val)
        np.save("data/processed/y_val.npy", y_val)

    if return_data:
        return X_train, X_val, y_train, y_val

if __name__ == "__main__":

    create_augmented_dataset(save=True)

