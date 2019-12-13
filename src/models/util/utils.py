import numpy as np
import json
import itertools
import random
import os
import datetime

from src.helper import get_config

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def log_model(model, c_backs, metrics_dic, config=None):
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
    log["metric"] = metrics_dic

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

def log_models_tries(log_dic, id):
    now = datetime.datetime.now()
    file_path = os.path.join('logs', now.strftime("log_{}_%Y_%m_%d_%H_%M.json".format(id)))

    with open(file_path, 'a+') as fp:
        json.dump(log_dic, fp, cls=NumpyEncoder)

def split_train_val(X, y, prc = 0.75, seed = None):


    rnd = np.random.RandomState(seed)
    
    n_pos = sum(y==True)
    n_neg = sum(y==False)
    total = len(y)

    X_pos = X[y==True]
    X_neg = X[y!=True]


    n_pos_train = int(n_pos*prc)
    n_neg_train = int(n_neg*prc)

    n_pos_val = n_pos - n_pos_train
    n_neg_val = n_neg - n_neg_train

    p = rnd.choice(n_pos,n_pos_train,replace= False)
    n = rnd.choice(n_neg, n_neg_train, replace=False)

    X_train = np.concatenate((X_pos[p], X_neg[n]))
    y_train = np.concatenate((np.full(shape= n_pos_train, fill_value=True), np.full(shape= n_neg_train, fill_value=False)))

    X_val = np.concatenate((np.delete(X_pos, p, 0), np.delete(X_neg, n, 0)))
    y_val = np.concatenate((np.full(shape= n_pos_val, fill_value=True), np.full(shape= n_neg_val, fill_value=False)))

    return X_train, X_val, y_train, y_val

def load_architecture(arch):

    from src.models.architectures.FCCN_ import FCCN
    from src.models.architectures.VGG16_ import VGG16
    from src.models.architectures.XMASNET import XmasNet
    from src.models.architectures.ALEXNET import AlexNet
    from src.models.architectures.ResNet import ResNet
    from src.models.architectures.DenseNet import DenseNet
    from src.models.architectures.CRFNNVGG import CRFNNVGG
    from src.models.architectures.CRFXmasNet import CRFXmasNet
    from src.models.architectures.CRFResNet import CRFResNet
    from src.models.architectures.VCRF import VCRF
    from src.models.architectures.CRFAlexNet import CRFAlexNet

    arch = arch.lower()
    if arch == "fccn":
        return FCCN()
    elif arch == "vgg16":
        return VGG16()
    elif arch == "xmasnet":
        return XmasNet()
    elif arch == "alexnet":
        return AlexNet()
    elif arch == "resnet":
        return ResNet()
    elif arch == "densenet":
        return DenseNet()
    elif arch == "crfvgg":
        return CRFNNVGG()
    elif arch == "crfxmasnet":
        return CRFXmasNet()
    elif arch == "crfresnet":
        return CRFResNet()
    elif arch == "vcrf":
        return VCRF()
    elif arch == "crfalex":
        return CRFAlexNet()
    else:
        print("Arch: {} is not valid. Returning fccn.".format(arch))
        return FCCN()

def reservoir_sample(iterable, k):
    it = iter(iterable)
    if not (k > 0):
        raise ValueError("sample size must be positive")

    sample = list(itertools.islice(it, k)) # fill the reservoir
    random.shuffle(sample) # if number of items less then *k* then
                           #   return all items in random order.
    for i, item in enumerate(it, start=k+1):
        j = random.randrange(i) # random [0..i)
        if j < k:
            sample[j] = item # replace item with gradually decreasing probability
    return sample

def gen_combinations(d):
    keys, values = d.keys(), d.values()
    combinations = itertools.product(*values)

    for c in combinations:
        yield dict(zip(keys, c))

def rename_file(src_file, dst_file):
    import shutil
    shutil.copy(src_file, dst_file)

def calculate_roc(model, X, y):
    from sklearn.metrics import roc_auc_score

    y_pred = model.predict(X)

    return float(roc_auc_score(y, y_pred))

def calculate_cross_entropy(model, y_true, X_pred):
    from keras.metrics import binary_crossentropy
    from keras import backend as K

    y_true = np.asarray(y_true).astype('float32').reshape((-1,1))
    y_true = K.variable(y_true)

    y_pred = model.predict(X_pred)
    y_pred = np.asarray(y_pred).astype('float32').reshape((-1,1))
    y_pred = K.variable(model.predict(X_pred))

    error = K.eval(binary_crossentropy(y_true, y_pred))

    mean_error = float(np.mean(error))

    return mean_error

def reshape_flat_array(data):
    y = data[:, 4096]
    X = data[:, :4096]
    X = X.reshape((-1, 64, 64, 1))

    return X, y

def list_of_features_files():
    feat_path = "data/processed/intermediate/"
    runs_files = os.listdir(feat_path)
    list_of_runs = list(set(["_".join(a.split("_")[:3]) for a in runs_files]))
    list_of_runs.sort()
    return list_of_runs


