from keras.applications.densenet import DenseNet121
from src.models.architectures.base_architecture import BaseArchitecture


from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout

from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

class DenseNet(BaseArchitecture):
    name = "DenseNet"
    def architecture(self):
        base_model = DenseNet121(include_top=False, input_shape=(self.input_shape), classes=1, weights='imagenet')
        
        top_model = Sequential()

        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(.5)(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        return model