import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_metric(y_pred, y_true):

    y_pred_ = np.argmax(y_pred, axis=1)
    y_true_ = np.argmax(y_true, axis=1)
    
    return {
        "confusion_matrix": confusion_matrix(y_true=y_true_, y_pred=y_pred_),
        "accuracy_score": accuracy_score(y_true=y_true_, y_pred=y_pred_),
        "precision_score": precision_score(y_true=y_true_, y_pred=y_pred_, average='micro'),
        "recall_score": recall_score(y_true=y_true_, y_pred=y_pred_, average='micro'),
        "f1_score": f1_score(y_true=y_true_, y_pred=y_pred_, average='micro')
    }
    