import numpy as np
import yaml 

import keras
from keras.metrics import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf



from src.helper import get_config
from src.models.util.callbacks import Metrics


from src.models.util.optimizers import load_optimizer
from src.models.util.callbacks import load_callbacks
# from src.models.util.utils import split_train_val, log_model, load_architecture, reservoir_sample, rename_file, gen_combinations
from src.models.util.utils import *
from src.features.build_features import apply_transformations, create_augmented_dataset, apply_rescale
# Image size: 256, 256, 1
# 1, 2, 8, 16, 32, 64, 128, 256, 512


def train_model(current = 0):
    # X_train = np.load("data/processed/X_train.npy")
    # X_val = np.load("data/processed/X_val.npy")

    # y_train  = np.load("data/processed/y_train.npy")
    # y_val  = np.load("data/processed/y_val.npy")

    X = np.load("data/processed/X_36.npy")
    y = np.load("data/processed/y_36.npy")

    X_train, X_val, y_train, y_val = create_augmented_dataset(X = X, y = y, return_data=True)

    X_train = apply_rescale(X_train)
    X_val = apply_rescale(X_val)

    config = get_config()
    arc = load_architecture(config["train"]["optimizers"]["architecture"])
    model = arc.architecture()

    # model.summary()

    print(" ################################## ")
    print(" #      MODEL CONFIGURATION       # ")

    print("CURRENT MODEL: " + str(current))
    for k in config["train"]["optimizers"]:
        print( str(k) + ":" + str(config["train"]["optimizers"][k]))
    print(" ################################## ")



    c_backs = load_callbacks(arc.weights_path)
    opt = load_optimizer()

    model.compile( optimizer=opt, loss='binary_crossentropy')


    if(config["train"]["use_augmentation"]):
        train_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        train_datagen.fit(X_train)

        train_generator = train_datagen.flow(X_train, y_train, batch_size=config["train"]["batch_size"])

        val_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
        )

        val_datagen.fit(X_val)

        val_generator = val_datagen.flow(X_val, y_val, batch_size=config["train"]["batch_size"])

        model.fit_generator(train_generator,
                        steps_per_epoch=len(X_train) / config["train"]["batch_size"],
                        epochs=config["train"]["epochs"],
                        validation_data=val_generator,
                        validation_steps = len(X_val) / config["train"]["batch_size"],
                        shuffle=True,
                        callbacks=c_backs )
    else:
        model.fit(X_train, y_train,
                batch_size=config["train"]["batch_size"],
                epochs=config["train"]["epochs"],
                validation_data=(X_val, y_val),
                shuffle=True,
                callbacks=c_backs)

    log_model(model, c_backs, config)

if __name__ == "__main__":
    grid_search = True

    if grid_search == False:
        train_model()

    else:
        config = get_config()
        with open("search_params.yml", 'r') as ymlfile:
            grid_search = yaml.load(ymlfile)

        rename_file('config.yml', 'config.yml.default')

        sample = reservoir_sample(gen_combinations(grid_search["train"]["optimizers"]), 10)

        i = 0

        for s in sample:
            i = i + 1

            for key in s:
                if key in config["train"]["optimizers"].keys():
                    config["train"]["optimizers"][key] = s[key]

            with open('config.yml', 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

                try:
                    train_model(current = i)
                    rename_file('config.yml.default', 'config.yml')
                except:
                    print("Error occured")