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
# Image size: 256, 256, 1
# 1, 2, 8, 16, 32, 64, 128, 256, 512


if __name__=="__main__":

    config = get_config()

    arc = load_architecture(config["train"]["architecture"])
    model = arc.architecture()

    model.summary()

    X = np.load("data/processed/X_2c.npy")
    y = np.load("data/processed/y_2c.npy")

    train_samples = round(X.shape[0] * config["train"]["train_val_split"])

    X_train = X[:train_samples,: ,:,:]
    X_val = X[train_samples:, :,::]

    y_train = y[:train_samples]
    y_val = y[train_samples:]

    # X_train, X_val, y_train, y_val = split_train_val(X, y, config["train"]["train_val_split"], seq=True)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(X_train)

    c_backs = load_callbacks(arc.weights_path)
    opt = load_optimizer()

    model.compile( optimizer=opt, loss='binary_crossentropy')

    # model.fit_generator(datagen.flow(X_train, y_train, batch_size=config["train"]["batch_size"]),
    #                 steps_per_epoch=len(X_train) / config["train"]["batch_size"],
    #                 epochs=config["train"]["epochs"],
    #                 validation_data=(X_val, y_val),
    #                 shuffle=True,
    #                 callbacks=c_backs )


    model.fit(X_train, y_train,
            batch_size=config["train"]["batch_size"],
            epochs=config["train"]["epochs"],
            validation_data=(X_val, y_val),
            shuffle=True,
            callbacks=c_backs)

    log_model(model, c_backs, config)