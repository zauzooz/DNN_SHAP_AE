import numpy as np
import pandas as pd

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}_{x}"

        df[dummy_name] = dummies[x]

        df[dummy_name] *= 1

    df = df.drop(name, axis=1, inplace=True)

    return df
