from keras.applications.resnet50 import ResNet50
from src.models.architectures.base_architecture import BaseArchitecture


class ResNet(BaseArchitecture):
    name = "ResNet"
    def architecture(self):

        model = ResNet50(include_top=False, input_shape=self.input_shape, classes=2, weights=None)
        return model