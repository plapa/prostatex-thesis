from keras.applications.inception_resnet_v2 import InceptionResNetV2 as IncepRes

from src.models.architectures.base_architecture import *

class InceptionResNetV2(BaseArchitecture):
    name = "InceptionResNetV2"
    def architecture(self):
        model = IncepRes(include_top=False, input_shape = self.input_shape, classes= 2)
        return model