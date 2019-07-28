

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, Flatten, Dense, BatchNormalization, Activation

from src.models.architectures.layers import CrfRnnLayer
from src.models.architectures.base_architecture import BaseArchitecture

from src.helper import get_config


class CRFXmasNet(BaseArchitecture):

    name = "CRFXmasNet"
    flaten_layer = "flatten_1"
    def architecture(self):

        config = get_config()
        config = config["train"]["optimizers"]

        channels, height, weight = 3, 500, 500

        # Input
        #input_shape = (height, weight, 3)
        img_input = Input(shape=self.input_shape)
        #img_input = Cropping2D((3,3))(img_input)

        # Add plenty of zero padding
        #x = ZeroPadding2D(padding=(218, 218))(img_input)

        # block 1

        x = Conv2D(strides=1, filters= 32, kernel_size=3, padding="same")(img_input)
        x = BatchNormalization()(x)
        x = Activation(activation="relu")(x)
        x = Conv2D(strides=1, filters= 32, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation(activation="relu")(x)

        block1 = x




        x = MaxPooling2D((2,2), 2, padding="same")(x)


        # block 2

        x = Conv2D(strides=1, filters= 32, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation(activation="relu")(x)

        x = Conv2D(strides=1, filters= 32, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation(activation="relu")(x)

        block2 = x


        x = MaxPooling2D((2,2), 2, padding="same")(x)
        

        x = Conv2D(512, (7, 7), activation='relu', padding='valid', name='fc6')(x)
        x = Conv2D(512, (1, 1), activation='relu', padding='valid', name='fc7')(x)
        x = Conv2D(21, (1, 1), padding='valid', name='score-fr')(x)

        block3 = x

        # Deconvolution

        score1 = Conv2D(1, (1, 1))(block1)

        score2 = Conv2D(1, (1, 1))(block2)
        score2 = Conv2DTranspose(1, (2, 2), strides=2, name='score2', use_bias=False)(score2)

        score3 = Conv2D(1, (1, 1))(block3)
        score3 = Conv2DTranspose(1, (2, 2), strides=6)(score3)
        #
        score3 = BatchNormalization()(score3)
        score3 = ZeroPadding2D(2)(score3)


        # Fuse things together
        #score_final = Add()([score4, score_pool3c])

        # Final up-sampling and cropping

        score = Add()([score1, score2])
        #
        score = BatchNormalization()(score)
        score_pool = Add()([score, score3])
        #
        score_pool = BatchNormalization()(score_pool)

        output = CrfRnnLayer(image_dims=(64, 64),
                            num_classes=2,
                            theta_alpha= config["crf_theta_alpha"], #3
                            theta_beta= config["crf_theta_beta"], #3
                            theta_gamma= config["crf_theta_gamma"], #3
                            num_iterations= config["crf_num_iterations"],
                            name='crfrnn')([score_pool, img_input])
        #
        output = BatchNormalization()(output)

        classi = Add()([score_pool, output])
        k = Flatten()(classi)

        k = Dense(128, activation='relu')(k)
        k = Dropout(.5)(k)
        k = Dense(256, activation='relu')(k)
        predictions = Dense(1, activation='sigmoid')(k)

        # Build the model
        model = Model(img_input, predictions, name='crfrnn_net')

        return model

