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
#from src.features.build_features import create_augmented_dataset


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
set_session(tf.Session(config=config))

config = get_config()
global current 
current = 0
X = np.load("data/processed/X_t2_PD_ADC.npy")
y = np.load("data/processed/y_t2_PD_ADC.npy")

#X_train, X_test, y_train, y_test = split_train_val(X = X, y = y, seed = 42)

X_train, X_test, y_train, y_test = split_train_val(X = X, y = y, prc = .8, seed = 42)

X_train, X_val, y_train, y_val = split_train_val(X = X_train, y = y_train, prc = .75, seed = 42)

if(config["preprocessing"]["use_augmentation"]):
    X_train, y_train = create_augmented_dataset(X_train, y_train, return_data=True)


X_train = np.concatenate((X_train, X_val))
y_train = np.concatenate((y_train, y_val))
X_train = apply_rescale(X_train)
X_test = apply_rescale(X_test)
X_val = apply_rescale(X_val)


def train_model():
    logs = pd.DataFrame()
    print(" ################################## ")
    print(" #      MODEL CONFIGURATION       # ")

    print("CURRENT MODEL: " + str(current))
    for k in config["train"]["optimizers"]:
        print( str(k) + ":" + str(config["train"]["optimizers"][k]))
    print(" ################################## ")

    for experiment_run in range(config["train"]["n_runs"]):

        start = datetime.datetime.now()

        print("CURRENT MODEL: " + str(current))
        print("EXPERIMENT: {}".format(experiment_run))
        print("TIME: {}".format(start))

        arc = load_architecture(config["train"]["optimizers"]["architecture"])
        model = arc.architecture()
        c_backs = load_callbacks(arc.weights_path)
        opt = load_optimizer()



        _loss = 'binary_crossentropy'

        model.compile( optimizer=opt, loss=_loss)
        start = datetime.datetime.now()


        if config["train"]["fit_model"]:
            model.fit(X_train, y_train,
                    batch_size=config["train"]["batch_size"],
                    epochs=config["train"]["epochs"],
                    validation_data=(X_val, y_val),
                    shuffle=True,
                    callbacks=c_backs,
                    verbose = config["train"]["verbose"])

        end = datetime.datetime.now()
        dif = end - start



        if config["train"]["fit_model"]:
            best_model = arc.architecture()
            best_model.load_weights(arc.weights_path)
        else:
            best_model = model


        roc_train = calculate_roc(best_model, X_train, y_train)
        roc_val = calculate_roc(best_model, X_val, y_val)

        val_loss = calculate_cross_entropy(best_model, y_val, X_val)

        b_train_loss = calculate_cross_entropy(best_model, y_train, X_train)
        b_val_loss = val_loss
        b_test_loss = calculate_cross_entropy(best_model, y_test, X_test)

        b_train_roc = roc_train
        b_val_roc = roc_val
        b_test_roc = calculate_roc(best_model, X_test, y_test)


        if config["train"]["save_intermediate_outputs"]:
            label = "best_{}_{}".format(arc.name, experiment_run)
            save_results_wrapper(best_model,  'flatten_1', label, X_train, y_train, X_test, y_test, X_val, y_val)

        log_row = {"id" : experiment_run, "metrics.time" : dif.total_seconds(), "metrics.loss.train":  b_train_loss, "metrics.loss.test": b_test_loss, "metrics.loss.val": b_val_loss, "metrics.roc.train" : b_train_roc, "metrics.roc.test" : b_test_roc, "metrics.roc.val" : b_val_roc}

        for key, value in config["train"]["optimizers"].items():
            log_row[key] = value
        logs = logs.append(log_row, ignore_index=True)


        print("EXPERIMENT: {}".format(experiment_run))
        print("TRAIN LOSS {}".format(b_train_loss))
        print("VAL LOSS {}".format(b_val_loss))
        print("TEST LOSS {}".format(b_test_loss))
        print("TEST AUROC {}".format(b_test_roc))


        log_model(model, c_backs, log_row)
        K.clear_session()

    now = datetime.datetime.now()

    log_path = os.path.join('outputs', 'logs', now.strftime("log_{}_%Y_%m_%d_%H_%M.csv".format(arc.name)))
    logs.to_csv(log_path)

    print( "LOGS SAVED AS: {}: ".format(log_path))

    print(" ")
    print(" ")
    print(" ")
    print(" ")

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
if __name__ == "__main__":
    search()


    # def set_f_path(aa):
    #     return "data/processed/{}.npy".format(aa)
    # np.save(set_f_path("X_train"), X_train)
    # np.save(set_f_path("X_test"), X_test)
    # np.save(set_f_path("X_val"), X_val)
    # np.save(set_f_path("y_train"), y_train)
    # np.save(set_f_path("y_test"), y_test)
    # np.save(set_f_path("y_val"), y_val)