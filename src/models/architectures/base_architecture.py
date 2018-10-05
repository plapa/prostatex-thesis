import os
from src.helper import get_config

class BaseArchitecture():
    weights_directory= "data/interim/weights/"

    name = "Base Architecture"

    config = get_config()
    input_shape = (2 * config["general"]["padding"], 2 * config["general"]["padding"], config["general"]["channels"] )
    def __init__(self, load_weights = False):
        self.id = self.name.lower().strip().replace(" ","_")
        self.file = "{}.h5".format(self.id)
        self.weights_path = os.path.join(self.weights_directory, self.file)
        self.load_weights = load_weights

    def architecture(self):
        ''' This method returns the keras architecture of the model. It should be implemented by each sub class '''
        raise NotImplementedError
