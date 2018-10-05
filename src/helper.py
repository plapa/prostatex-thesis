
import yaml

def get_config():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return cfg


conf = get_config()

print(conf)