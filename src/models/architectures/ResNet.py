from keras.applications.resnet50 import ResNet50
from src.models.architectures.base_architecture import BaseArchitecture
from keras.layers import Input

class ResNet(BaseArchitecture):
    name = "ResNet"
    def architecture(self):
        model = ResNet50(include_top=False, input_shape=(64,64,3), classes=2, weights=None)
        return model