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
from src.models.util.intermediate_output import save_results_wrapper
from src.models.util.utils import *
from src.features.build_features import  create_augmented_dataset, apply_rescale


config = get_config()
global current 
current = 0

X = np.load("data/processed/X_t2_PD_ADC.npy")
y = np.load("data/processed/y_t2_PD_ADC.npy")

#X_train, X_test, y_train, y_test = split_train_val(X = X, y = y, seed = 42)

X_train, X_test, y_train, y_test = split_train_val(X = X, y = y, prc = .8, seed = 42)

X_train, X_val, y_train, y_val = split_train_val(X = X_train, y = y_train, prc = .75, seed = 42)

if(config["train"]["use_augmentation"]):
    X_train, y_train = create_augmented_dataset(X_train, y_train, return_data=True)

X_train = apply_rescale(X_train)
X_test = apply_rescale(X_test)
X_val = apply_rescale(X_val)


def train_model():

    print("#####################################")

    print(" ################################## ")
    print(" #      MODEL CONFIGURATION       # ")

    print("CURRENT MODEL: " + str(current))
    for k in config["train"]["optimizers"]:
        print( str(k) + ":" + str(config["train"]["optimizers"][k]))
    print(" ################################## ")

    for _ in range(config["train"]["n_runs"]):

        arc = load_architecture(config["train"]["optimizers"]["architecture"])
        model = arc.architecture()
        c_backs = load_callbacks(arc.weights_path)
        opt = load_optimizer()

        model.compile( optimizer=opt, loss='binary_crossentropy')
        model.fit(X_train, y_train,
                batch_size=config["train"]["batch_size"],
                epochs=config["train"]["epochs"],
                validation_data=(X_val, y_val),
                shuffle=True,
                callbacks=c_backs,
                verbose = 1)

        best_model = arc.architecture()
        best_model.load_weights(arc.weights_path)


        roc_train = calculate_roc(best_model, X_train, y_train)
        roc_val = calculate_roc(best_model, X_val, y_val)

        val_loss = calculate_cross_entropy(best_model, y_val, X_val)

        b_train_loss = calculate_cross_entropy(best_model, y_train, X_train)
        b_val_loss = val_loss
        b_test_loss = calculate_cross_entropy(best_model, y_test, X_test)

        b_train_roc = roc_train
        b_val_roc = roc_val
        b_test_roc = calculate_roc(best_model, X_test, y_test)

        best_config = dict()
        best_config["params"] = config["train"]["optimizers"].copy()

        best_config["metrics"] = dict()
        best_config["metrics"]["roc"] = dict()
        best_config["metrics"]["roc"]["train"] = b_train_roc
        best_config["metrics"]["roc"]["test"] = b_test_roc
        best_config["metrics"]["roc"]["val"] =  b_val_roc

        best_config["metrics"]["loss"] = dict()
        best_config["metrics"]["loss"]["train"] = b_train_loss
        best_config["metrics"]["loss"]["test"] = b_test_loss
        best_config["metrics"]["loss"]["val"] =  b_val_loss
        print("TRAIN LOSS {}".format(b_train_loss))
        print("VAL LOSS {}".format(b_val_loss))
        print("TEST LOSS {}".format(b_test_loss))

        #save_results_wrapper(best_model, best_config, 'flatten_2', arc.name , X_train, y_train, X_test, y_test, X_val, y_val)
        log_models_tries(best_config)
        # print("ROC TRAIN: {}".format(roc_train))
        # print("ROC VAL: {}".format(roc_val))

        # metric_dick = {"AUROC" : {"TRAIN" : roc_train, "VAL" : roc_val}}
        # log_model(model, c_backs, config)
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

        sample = reservoir_sample(gen_combinations(grid_search["train"]["optimizers"]), config["train"]["gridsearch_number_tries"])


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

