from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation


from src.models.architectures.base_architecture import *

class FCCN(BaseArchitecture):
    name = "FCCN"
    def architecture(self):
        model = Sequential()

        model.add(Conv2D(input_shape= self.input_shape, strides=1, filters= 64, kernel_size=3, padding="same"))
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