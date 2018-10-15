
import yaml

def get_config():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return cfg

def _load_config_params(params_to_parse, blank_opt, opt_config):
    if opt_config["use_default_params"]:
        opt = blank_opt()
    else:

        params = dict()

        for key in params_to_parse:
            if key in opt_config.keys():
                params[key] = opt_config[key]

        opt = blank_opt(**params)

    return opt