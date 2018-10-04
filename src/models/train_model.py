import keras
import numpy as np

from keras.metrics import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from architectures.vgg16 import VGG_16

# Image size: 256, 256, 1
# 1, 2, 8, 16, 32, 64, 128, 256, 512


if __name__=="__main__":
    model = VGG_16()

    model.summary()

    X = np.load("../data/processed/X_2c.npy")
    y = np.load("../data/processed/y_2c.npy")

    np.mean(X)
    np.std(X)

   #  X = (X - np.mean(X)) / np.std(X)

    train_samples = 200

    X_train = X[:train_samples,: ,:,:]
    X_val = X[train_samples:, :,::]


    y_train = y[:train_samples]
    y_val = y[train_samples:]

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    #datagen.fit(X_train)


    model_checkpoint = ModelCheckpoint('../data/weights.h5', monitor='loss', save_best_only=True)


    c_backs = [model_checkpoint]
    c_backs.append( EarlyStopping(monitor='loss', min_delta=0.001, patience=5) )

    model.compile( optimizer=SGD(lr=0.001, momentum=0.00005, nesterov=True), loss='binary_crossentropy')

    # model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
    #                 steps_per_epoch=len(X_train) / 32, epochs=200,validation_data=(X_val, y_val),
    #         shuffle=True,
    #         callbacks=c_backs )


    model.fit(X_train, y_train,
            batch_size=10,
            epochs=200,
            validation_data=(X_val, y_val),
            shuffle=True,
            callbacks=c_backs)