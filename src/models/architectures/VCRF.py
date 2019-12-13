

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, Flatten, Dense, BatchNormalization, Activation

from src.models.architectures.layers import CrfRnnLayer
from src.models.architectures.base_architecture import BaseArchitecture

from src.helper import get_config


class VCRF(BaseArchitecture):

    name = "VCRF"
    flaten_layer = "flatten_1"
    def architecture(self):

        config = get_config()
        config = config["train"]["optimizers"]

        channels, height, weight = 3, 500, 500

        # Input
        #input_shape = (height, weight, 3)
        img_input = Input(shape=(64,64,1))

        output = CrfRnnLayer(image_dims=(64, 64),
                            num_classes=1,
                            theta_alpha= config["crf_theta_alpha"], #3
                            theta_beta= config["crf_theta_beta"], #3
                            theta_gamma= config["crf_theta_gamma"], #3
                            num_iterations= config["crf_num_iterations"],
                            name='crfrnn')([img_input, img_input])
        #

        k = Flatten()(output)
        predictions = Dense(1, activation='sigmoid')(k)

        model = Model(img_input, predictions, name='crfrnn_net')

        return model

