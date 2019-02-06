from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D


from src.models.architectures.base_architecture import *

class AlexNet(BaseArchitecture):
    name = "AlexNet"


    def architecture(self):

	    # Initialize model
        alexnet = Sequential()

        # Layer 1
        alexnet.add(Conv2D(96, (11, 11), input_shape=self.input_shape,
            padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPool2D(pool_size=(2, 2)))

        # Layer 2
        alexnet.add(Conv2D(256, (5, 5), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPool2D(pool_size=(2, 2)))

        # Layer 3
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(512, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPool2D(pool_size=(2, 2)))

        # Layer 4
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(512, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))

        # Layer 5
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(512, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPool2D(pool_size=(2, 2)))

        # Layer 6
        alexnet.add(Flatten())
        alexnet.add(Dense(512))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.5))

        # Layer 7
        alexnet.add(Dense(512))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.5))

        # Layer 8
        alexnet.add(Dense(1))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('sigmoid'))

        return alexnet


