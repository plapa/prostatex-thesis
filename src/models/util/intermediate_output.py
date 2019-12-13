from keras.models import Model
from src.helper import get_config
import numpy as np
import os
import yaml

config = get_config()

def get_intermediate_output(model, layer_name, X, y = None):
    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(X)

    return intermediate_output

def save_results(model, X, y, label, d_set, layer_name):
    out_folder = config["meta"]["layers_path"]
    weights_path = os.path.join(out_folder, "{}.h5".format(label))
    np_path =  os.path.join(out_folder, "{}_{}".format(label, d_set))

    x_ = get_intermediate_output(model, layer_name, X, y)
    output = np.column_stack((x_, y))

    #model.save(weights_path)
    np.save(np_path, output)


def save_results_wrapper(model, layer_name, label, X_train, y_train, X_test, y_test, X_val, y_val):
    print(model.summary())
    save_results(model=model, X= X_train, y = y_train,layer_name = layer_name,label = label, d_set = "train")    
    save_results(model = model,X=  X_test,y= y_test, layer_name= layer_name,label = label,d_set= "test")
    save_results(model = model,X=  X_val,y= y_val, layer_name= layer_name,label = label,d_set= "val")

    X_train_val = np.concatenate((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    save_results(model = model,X= X_train_val,y= y_train_val, layer_name= layer_name,label = label,d_set= "train_val")
