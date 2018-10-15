import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def log_model(model, c_backs, config=None):
    """ Description
    :type model: Keras model
    :param model:

    :type c_backs: Keras call_back file
    :param c_backs:

    :type config: A dict containing the current parameters
    :param config:

    :raises:

    :rtype:
        """

    import datetime

    if config is None: 
        from src.helper import get_config

        config = get_config()


    now = datetime.datetime.now()

    log = dict()
    log["datetime"] = {'date': now.strftime("%Y-%m-%d %H:%M"), 'unix' : now.timestamp()}

    hist = model.history.history
    hist["epoch"] = model.history.epoch


    for cb in c_backs:
        if cb.__class__.__name__ == 'Metrics':
                hist["val_auroc"] = cb.val_auroc
                hist["val_precision"] = cb.val_precisions
                hist["val_recall"] = cb.val_recalls
                
    log["parameters"] = config
    log["history"] = hist

    import os
    file_path = os.path.join('logs', now.strftime("log_%Y_%m_%d_%H_%M.json"))



    with open(file_path, 'a+') as fp:
        json.dump(log, fp, cls=NumpyEncoder)


def split_train_val(X, y, frac = 0.75, seq = False):
    n_samples = y.shape[0]
    train_size = round(n_samples * frac)
    

    if not seq: 
        train_samples = np.random.choice(n_samples, size = train_size, replace=False)
        print(train_samples)

        X_train = X[train_samples,: ,:,:]
        X_val = X[~train_samples, :,:, :]

        y_train = y[train_samples]
        y_val = y[~train_samples]

    else:
        X_train = X[:train_size,: ,:,:]
        X_val = X[train_size:, :,::]

        y_train = y[:train_size]
        y_val = y[train_size:]

    print("X_val: {} X_train: {} y_val: {}, y_train: {}".format(X_val.shape, X_train.shape, y_val.shape, y_train.shape))

    return X_train, X_val, y_train, y_val

def load_architecture(arch):

    from src.models.architectures.FCCN_ import FCCN
    from src.models.architectures.VGG16_ import VGG16
    from src.models.architectures.XMASNET import XmasNet
    from src.models.architectures.ALEXNET import AlexNet

    arch = arch.lower()
    if arch == "fccn":
        return FCCN()
    elif arch == "vgg16":
        return VGG16()
    elif arch == "xmasnet":
        return XmasNet()
    elif arch == "alexnet":
        return AlexNet()
    else:
        Print("Arch: {} is not valid. Returning fccn.".format(arch))
        return FCCN()

