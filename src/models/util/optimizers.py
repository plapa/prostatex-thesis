

from keras.optimizers import *
from src.helper import get_config, _load_config_params

def load_optimizer():

    config = get_config()

    opt_config = config["train"]["optimizers"]

    opt_name = opt_config["use"]


    if(opt_name == "adam"):
        params_to_parse = ["lr", "beta_1", "beta_2", "epsilon", "decay", "amsgrad"]

        opt = _load_config_params(params_to_parse, Adam, opt_config)

    elif(opt_name == "sgd"):
        params_to_parse = ["lr", "rho", "epsilon", "decay"]
        opt = _load_config_params(params_to_parse, SGD, opt_config)

    elif(opt_name == "rmsprop"):
        params_to_parse = ["lr", "rho", "epsilon", "decay"]
        opt = _load_config_params(params_to_parse, SGD, opt_config)

        opt = _load_config_params(params_to_parse, RMSprop, opt_config)

    elif(opt_name == "nadam"):
        params_to_parse = ["lr", "beta_1", "beta_2", "epsilon"]

        opt = _load_config_params(params_to_parse, Nadam, opt_config)

    return opt



if __name__ == "__main__":
    opt = load_optimizer()

    print(opt.get_config())
