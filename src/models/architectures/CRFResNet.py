from keras.applications.resnet50 import ResNet50

from keras.models import Sequential, Model

from keras import backend as K

from keras.layers import Dense, Flatten, Dropout, Conv2DTranspose, Cropping2D, ZeroPadding2D


from src.models.architectures.layers import CrfRnnLayer
from src.models.architectures.base_architecture import BaseArchitecture


from src.helper import get_config


class CRFResNet(BaseArchitecture):
    name = "CRFResNet"
    def architecture(self):
        config = get_config()
        config = config["train"]["optimizers"]

        channels, height, weight = 3, 500, 500

        input_shape = (height, weight, 3)

        base_model = ResNet50(include_top=False, input_shape=(self.input_shape), classes=1, weights=None)

        x = base_model.output

        score2 = Conv2DTranspose(1, (12, 12), strides=2, name='score2')(x)


        # Final up-sampling and cropping
        upsample = Conv2DTranspose(1, (12, 12), strides=4, name='upsample', use_bias=False)(score2)
        #upsample = Cropping2D((12, 12))(upsample)

        img_input = base_model.input
        x = ZeroPadding2D(padding=(218, 218))(img_input)

        output = CrfRnnLayer(image_dims=(64, 64),
                            num_classes=1,
                            theta_alpha= config["crf_theta_alpha"], #3
                            theta_beta= config["crf_theta_beta"], #3
                            theta_gamma= config["crf_theta_gamma"], #3
                            num_iterations= config["crf_num_iterations"],
                            name='crfrnn')([upsample, img_input])


        x = Flatten()(output)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(.5)(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        return model