import numpy as np


import keras
from keras.metrics import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


from src.models.architectures.FCCN_ import FCCN
from src.models.architectures.VGG16_ import VGG16
from src.models.architectures.XMASNET import XmasNet
from src.helper import get_config
from src.models.util.callbacks import Metrics


from src.models.util.optimizers import load_optimizer
from src.models.util.callbacks import load_callbacks
from src.models.util.utils import split_train_val, log_model, load_architecture
# from src.features.build_features import apply_transformations
# Image size: 256, 256, 1
# 1, 2, 8, 16, 32, 64, 128, 256, 512


if __name__=="__main__":


    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")

    y_train  = np.load("data/processed/y_train.npy")
    y_val  = np.load("data/processed/y_val.npy")

    config = get_config()
    arc = load_architecture(config["train"]["architecture"])
    model = arc.architecture()

    model.summary()

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