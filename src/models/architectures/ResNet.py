from keras.applications.resnet50 import ResNet50
from src.models.architectures.base_architecture import BaseArchitecture


from keras.models import Sequential, Model
from keras.layers import Dense, Flatten

from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

class ResNet(BaseArchitecture):
    name = "ResNet"
    def architecture(self):
        base_model = ResNet50(include_top=False, input_shape=(64,64,3), classes=1, weights=None, pooling=None)
        
        top_model = Sequential()

        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        return model