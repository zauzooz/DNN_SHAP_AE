import pickle
import pandas as pd
import numpy as np
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.iterative_method import BasicIterativeMethod
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.carlini import CarliniL2Method
from art.attacks.evasion.deepfool import DeepFool

def generate_defense_set(
    clean: pd.DataFrame,
    fsgm: pd.DataFrame,
    bim: pd.DataFrame,
    cw_l2: pd.DataFrame,
    jsma: pd.DataFrame,
    deepfool: pd.DataFrame,
    selected_features: list,
    shap_importance_feature_values: list,
    top_n_features: int = 10,
    random_mode: bool = False
):    
    n_selected_features = selected_features[:top_n_features]
    values = shap_importance_feature_values[:top_n_features]
    dfs = [
        clean[n_selected_features],
        fsgm[n_selected_features],
        bim[n_selected_features],
        cw_l2[n_selected_features],
        jsma[n_selected_features],
        deepfool[n_selected_features]
        ]
    dfs = pd.concat(dfs, axis=0, ignore_index=True)
    labels = [1 for _ in range(dfs.shape[0])]
    labels[:clean.shape[0]] = [0] * clean.shape[0]
    labels = pd.DataFrame(
        {"label":labels}
    )
    
    return pd.concat([dfs, labels], axis=1).sample(frac=1, ignore_index=True)

def generate_testset(classifier, test_samples: pd.DataFrame, y_test_samples: pd.DataFrame, selected_features: list, n_features: int = 20, random_mode: bool= False):
    import random
    import copy
    full_features = test_samples.columns
    top_n_features = selected_features[:n_features]
    
    index_number_n_features = [list(full_features).index(num) for num in top_n_features]
    
    adversarial_algs = {
        "Normal": None,
        "FSGM": FastGradientMethod(estimator=classifier, eps=0.2),
        "BIM": BasicIterativeMethod(estimator=classifier, eps=0.2, max_iter=100, batch_size=32),
        "CW-L2": CarliniL2Method(classifier=classifier, max_iter=10),
        "JSMA": SaliencyMapMethod(classifier=classifier,theta=0.1,gamma=1, batch_size=1),
        "DeepFool": DeepFool(classifier=classifier, max_iter=100, epsilon=0.000001, nb_grads=10, batch_size=1),
    }
    return_df = test_samples.values
    y_return = []
    
    alg_namelist = list(adversarial_algs.keys())
    
    random.seed(42)
    
    for i in range(len(return_df)):
        if len(alg_namelist) == 0:
            alg_namelist = list(adversarial_algs.keys())
        random.shuffle(alg_namelist)
        alg = random.choice(alg_namelist)
        alg_namelist.remove(alg)
        print(f"alg {alg}")
        if alg != "Normal":
            ex = return_df[i]
            ex = ex.reshape(1,ex.shape[0])
            adversarial_algs[alg].generate(x=ex)
            return_df[i, index_number_n_features] = ex[0,index_number_n_features]
            y_return.append(1)
        else:
            y_return.append(0)
    
    return_df = pd.DataFrame(return_df, columns=full_features)
    
    return return_df, y_return