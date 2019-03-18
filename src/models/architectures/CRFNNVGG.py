from src.models.architectures.base_architecture import BaseArchitecture


from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, Flatten, Dense
from src.models.architectures.layers import CrfRnnLayer

from src.helper import get_config


class CRFNNVGG(BaseArchitecture):

    name = "CRFNN"
    flaten_layer = "flatten_1"
    def architecture(self):
        config = get_config()
        config = config["train"]["optimizers"]

        channels, height, weight = 3, 500, 500

        # Input
        input_shape = (height, weight, 3)
        img_input = Input(shape=self.input_shape)
        #img_input = Cropping2D((3,3))(img_input)

        # Add plenty of zero padding
        x = ZeroPadding2D(padding=(218, 218))(img_input)

        # VGG-16 convolution block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        # VGG-16 convolution block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', padding='same')(x)

        # VGG-16 convolution block output3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', padding='same')(x)
        pool3 = x

        # VGG-16 convolution block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', padding='same')(x)
        pool4 = x

        # VGG-16 convolution block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5', padding='same')(x)

        # Fully-connected layers converted to convolution layers
        x = Conv2D(128, (7, 7), activation='relu', padding='valid', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(128, (1, 1), activation='relu', padding='valid', name='fc7')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(21, (1, 1), padding='valid', name='score-fr')(x)

        # Deconvolution
        score2 = Conv2DTranspose(1, (4, 4), strides=2, name='score2')(x)

        # Skip connections from pool4
        score_pool4 = Conv2D(1, (1, 1), name='score-pool4')(pool4)
        score_pool4c = Cropping2D((5, 5))(score_pool4)
        score_fused = Add()([score2, score_pool4c])
        score4 = Conv2DTranspose(1, (4, 4), strides=2, name='score4', use_bias=False)(score_fused)

        # Skip connections from pool3
        score_pool3 = Conv2D(1, (1, 1), name='score-pool3')(pool3)
        score_pool3c = Cropping2D((9, 9))(score_pool3)
        score_pool3c = ZeroPadding2D(padding=((1,0), (1,0)))(score_pool3c)


        # Fuse things together
        score_final = Add()([score4, score_pool3c])

        # Final up-sampling and cropping
        upsample = Conv2DTranspose(1, (4, 4), strides=4, name='upsample', use_bias=False)(score_final)
        upscore = Cropping2D(((56, 56), (56, 56)))(upsample)
        upscore = Cropping2D(((4, 4), (4, 4)))(upscore)

        output = CrfRnnLayer(image_dims=(64, 64),
                            num_classes=1,
                            theta_alpha= config["crf_theta_alpha"], #3
                            theta_beta= config["crf_theta_beta"], #3
                            theta_gamma= config["crf_theta_gamma"], #3
                            num_iterations= config["crf_num_iterations"],
                            name='crfrnn')([upscore, img_input])


        k = MaxPooling2D((2,2), 2)(output)
        k = Flatten()(k)

        k = Dense(128, activation='relu')(k)
        k = Dropout(.5)(k)
        k = Dense(256, activation='relu')(k)
        predictions = Dense(1, activation='sigmoid')(k)

        # Build the model
        model = Model(img_input, predictions, name='crfrnn_net')

        return model

