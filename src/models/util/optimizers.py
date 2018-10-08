

from keras.optimizers import *
from src.helper import get_config

def load_optimizer():

    config = get_config()

    opt_config = config["train"]["optimizers"]

    opt_name = opt_config["use"]


    if(opt_name == "adam"):
        params_to_parse = ["lr", "beta_1", "beta_2", "epsilon", "decay", "amsgrad"]

        opt = _load_optimizer_configs(params_to_parse, Adam, opt_config)

    elif(opt_name == "sgd"):
        params_to_parse = ["lr", "rho", "epsilon", "decay"]
        opt = _load_optimizer_configs(params_to_parse, SGD, opt_config)

    elif(opt_name == "rmsprop"):
        params_to_parse = ["lr", "rho", "epsilon", "decay"]
        opt = _load_optimizer_configs(params_to_parse, SGD, opt_config)

        opt = _load_optimizer_configs(params_to_parse, RMSprop, opt_config)

    elif(opt_name == "nadam"):
        params_to_parse = ["lr", "beta_1", "beta_2", "epsilon"]

        opt = _load_optimizer_configs(params_to_parse, Nadam, opt_config)

    return opt

def _load_optimizer_configs(params_to_parse, blank_opt, opt_config):
    if opt_config["use_default_params"]:
        opt = blank_opt()
    else:

        params = dict()

        for key in params_to_parse:
            if key in opt_config.keys():
                params[key] = opt_config[key]

        opt = blank_opt(**params)

    return opt


if __name__ == "__main__":
    opt = load_optimizer()

    print(opt.get_config())
