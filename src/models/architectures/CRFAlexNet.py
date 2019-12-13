from keras.models import Model
from keras.layers.core import Flatten, Dense
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Input, ZeroPadding2D, Conv2DTranspose, Add

from src.models.architectures.layers import CrfRnnLayer
from src.helper import get_config

from src.models.architectures.base_architecture import *

class CRFAlexNet(BaseArchitecture):
    name = "CRFAlexNet"

    def architecture(self):
        config = get_config()
        config = config["train"]["optimizers"]
	    # Initialize model
        # Layer 1
        img_input = Input(shape=self.input_shape)

        x = Conv2D(96, (11, 11), input_shape=self.input_shape,
            padding='same', activation='relu')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        block_1 = Conv2D(1, (1,1))(x)

        # Layer 2
        x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        block_2 = Conv2D(1, (1,1))(x)


        # Layer 4
        x = ZeroPadding2D((1,1))(x)
        x = Conv2D(512, (3, 3),padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

        # Layer 5
        x = ZeroPadding2D((1,1))(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

        block_3 =Conv2D(1, (1,1))(x)
        block_3 = ZeroPadding2D((2,2))(block_3)

        block_1 = Conv2DTranspose(1, (1, 1), strides = 2, name = "block_1")(block_1)

        block_2 = Conv2DTranspose(1, (1, 1), strides = 8, name = "block_2")(block_2)

        block_3 = Conv2DTranspose(1, (1, 1), strides = 4, name = "block_3")(block_3)

        upscore = Add()([block_1, block_3, block_2])

        output = CrfRnnLayer(image_dims=(64, 64),
                            num_classes=1,
                            theta_alpha= config["crf_theta_alpha"], #3
                            theta_beta= config["crf_theta_beta"], #3
                            theta_gamma= config["crf_theta_gamma"], #3
                            num_iterations= config["crf_num_iterations"],
                            name='crfrnn')([upscore, img_input])


        classi = Add()([upscore, output])
        k = Flatten()(classi)

        k = Dense(128, activation='relu')(k)
        k = Dropout(.5)(k)
        k = Dense(256, activation='relu')(k)
        predictions = Dense(1, activation='sigmoid')(k)

        # Build the model
        model = Model(img_input, predictions, name='CRFALEXNET')


        return model