import keras
import numpy as np

from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.models import Sequential

from keras.metrics import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

# Image size: 256, 256, 1

# 1, 2, 8, 16, 32, 64, 128, 256, 512
import tensorflow as tf


def FCCN():

    model = Sequential()

    model.add(Conv2D(input_shape=(64,64,3), strides=1, filters= 64, kernel_size=3, padding="same"))
    model.add(Conv2D(3, 1, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))



    model.add(Conv2D(strides=1, filters= 64, kernel_size=3, padding="same"))
    model.add(Conv2D(3, 1, padding= "same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))

    model.add(Conv2D(strides=1, filters= 128, kernel_size=3, padding= "same"))
    model.add(Conv2D(3, 1, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))

    model.add(Conv2D(strides=1, filters= 128, kernel_size=3, padding= "same"))
    model.add(Conv2D(3, 1, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    
    model.add(Flatten())


    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    #model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    return model

if __name__=="__main__":
    model = FCCN()

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
    c_backs.append( EarlyStopping(monitor='loss', min_delta=0.001, patience=10) )

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