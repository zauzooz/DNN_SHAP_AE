import pandas as pd
import numpy as np

def shap_importance(columns, shap_val_of_sample):
    importance_sample = pd.DataFrame([columns, np.abs(shap_val_of_sample).mean(axis=0).sum(axis=0).tolist()]).T
    importance_sample.columns = ['column_name', 'shap_importance']
    importance_sample = importance_sample.sort_values('shap_importance', ascending=False)
    return importance_sample