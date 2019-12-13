import numpy as np
import yaml 

import keras
from keras.metrics import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as K
import pandas as pd
import datetime

from src.helper import get_config
from src.models.util.callbacks import Metrics


from src.models.util.optimizers import load_optimizer
from src.models.util.callbacks import load_callbacks
from src.models.util.intermediate_output import save_results_wrapper
from src.models.util.utils import *
from src.features.build_features import apply_rescale



config = get_config()


arc = load_architecture(config["train"]["optimizers"]["architecture"])
model = arc.architecture()

print(model.summary())
c_backs = load_callbacks(arc.weights_path)
opt = load_optimizer()

_loss = 'binary_crossentropy'
