from src.models.architectures.base_architecture import BaseArchitecture

from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation


class XmasNet(BaseArchitecture):

    name = "XmasNet"
    def architecture(self):
        model = Sequential()

        model.add(MaxPool2D(2, input_shape=self.input_shape))
        
        model.add(Conv2D(strides=1, filters= 32, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        
        model.add(Conv2D(strides=1, filters= 32, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPool2D((2,2), 2, padding="same"))
        
        model.add(Conv2D(strides=1, filters= 64, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2D(strides=1, filters= 64, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPool2D((2,2), 2, padding="same"))

        model.add(Flatten())

        model.add(Dense(1024))

        model.add(Activation('relu'))

        model.add(Dense(256))

        model.add(Activation('relu'))
        model.add(Dense(1, activation = 'sigmoid'))

        return model


