import numpy as np
import keras
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

class Metrics(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auroc = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        _val_f1 = round(f1_score(val_targ, val_predict), 4)
        _val_recall = round(recall_score(val_targ, val_predict), 4)
        _val_precision = round(precision_score(val_targ, val_predict), 4)
        _val_auroc = round(roc_auc_score(val_targ, val_predict), 4)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auroc.append(_val_auroc)
        print("- val_f1: {} - val_precision: {} - val_recall {} - val_auroc {} ".format(_val_f1, _val_precision, _val_recall, _val_auroc))
        return
    
