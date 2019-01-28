import numpy as np
import yaml 

import keras
from keras.metrics import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as K
from src.helper import get_config
from src.models.util.callbacks import Metrics


from src.models.util.optimizers import load_optimizer
from src.models.util.callbacks import load_callbacks

from src.models.util.utils import *
from src.features.build_features import  create_augmented_dataset, apply_rescale
# Image size: 256, 256, 1
# 1, 2, 8, 16, 32, 64, 128, 256, 512

config = get_config()
global current 
current = 0

def train_model():
    global current

    X = np.load("data/processed/X_tf_t2_kt.npy")
    y = np.load("data/processed/y_tf_t2_kt.npy")

    X_train, X_val, y_train, y_val = split_train_val(X = X, y = y, seed = 42)

    if(config["train"]["use_augmentation"]):
        X_train, y_train = create_augmented_dataset(X_train, y_train, return_data=True)

    X_train = apply_rescale(X_train)
    X_val = apply_rescale(X_val)


    arc = load_architecture(config["train"]["optimizers"]["architecture"])
    model = arc.architecture()

    print(" ################################## ")
    print(" #      MODEL CONFIGURATION       # ")

    print("CURRENT MODEL: " + str(current))
    for k in config["train"]["optimizers"]:
        print( str(k) + ":" + str(config["train"]["optimizers"][k]))
    print(" ################################## ")

    c_backs = load_callbacks(arc.weights_path)
    opt = load_optimizer()

    model.compile( optimizer=opt, loss='binary_crossentropy')
    model.fit(X_train, y_train,
            batch_size=config["train"]["batch_size"],
            epochs=config["train"]["epochs"],
            validation_data=(X_val, y_val),
            shuffle=True,
            callbacks=c_backs)

    best_model = arc.architecture()
    best_model.load_weights(arc.weights_path)


    roc_train = calculate_roc(best_model, X_train, y_train)
    roc_val = calculate_roc(best_model, X_val, y_val)
    metric_dick = {"AUROC" : {"TRAIN" : roc_train, "VAL" : roc_val}}
    log_model(model, c_backs, config)



    print("ROC TRAIN: {}".format(roc_train))
    print("ROC VAL: {}".format(roc_val))

    K.clear_session()


def search():

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config = tf_config)

    tf.logging.set_verbosity(tf.logging.ERROR)

    if config["train"]["use_gridsearch"] == False:
        train_model()
    else:
        global current
        with open("search_params.yml", 'r') as ymlfile:
            grid_search = yaml.load(ymlfile)

        rename_file('config.yml', 'config.yml.default')

        sample = reservoir_sample(gen_combinations(grid_search["train"]["optimizers"]), 10)


        for s in sample:
            current = current + 1

            for key in s:
                if key in config["train"]["optimizers"].keys():
                    config["train"]["optimizers"][key] = s[key]

            with open('config.yml', 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

                train_model()
                rename_file('config.yml.default', 'config.yml')
                # try:
                #     train_model(current = i)
                #     rename_file('config.yml.default', 'config.yml')
                # except:
                #     print("Error occured")

if __name__ == "__main__":
    search()

