import pandas as pd
from evaluation_metric import evaluate_metric

def adversarial_n_best_worst_features(
    model,
    selected_features: list,
    samples: pd.DataFrame,
    adv_samples: pd.DataFrame,
    y_true
):
    columns = samples.columns
    acc_n_best_features = []
    acc_n_worst_features = []
    
    for i in range(len(selected_features)):
        # n best features
        n_best_features = selected_features[:i+1]
        static_features = [f for f in columns if f not in selected_features]
        adv_feature_val = adv_samples[n_best_features]
        the_adv_sample = samples.copy()
        for f in n_best_features:
            the_adv_sample[f] = adv_samples[f].values
        acc_n_best_features.append(evaluate_metric(
            y_pred=model.predict(the_adv_sample),
            y_true=y_true
            )['accuracy_score'])
        
        # n worst features
        n_worst_features = selected_features[i+1:]
        static_features = [f for f in columns if f not in selected_features]
        adv_feature_val = adv_samples[n_worst_features]
        the_adv_sample = samples.copy()
        for f in n_worst_features:
            the_adv_sample[f] = adv_samples[f].values
        acc_n_worst_features.append(evaluate_metric(
            y_pred=model.predict(the_adv_sample),
            y_true=y_true
            )['accuracy_score'])
        
    acc_n_worst_features.reverse()
    
    return (acc_n_best_features, acc_n_worst_features)

def plot_n_best_vs_n_worst(
    selected_features: list,
    acc_n_best_important_features,
    acc_n_worst_important_features
):
    import matplotlib.pyplot as plt
    plt.plot(
        [a+1 for a in range(len(selected_features))],
        acc_n_best_important_features, 'red')

    plt.plot(
        [a+1 for a in range(len(selected_features))],
        acc_n_worst_important_features , 'blue')

    plt.show()