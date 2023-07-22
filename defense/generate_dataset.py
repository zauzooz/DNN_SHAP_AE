import pandas as pd

def generate_dataset(
    clean: pd.DataFrame,
    fsgm: pd.DataFrame,
    bim: pd.DataFrame,
    cw_l2: pd.DataFrame,
    jsma: pd.DataFrame,
    deepfool: pd.DataFrame,
    selected_features: list,
    top_n_features: int = 10
):    
    n_selected_features = selected_features[:top_n_features]
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