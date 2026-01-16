from pathlib import Path

import pandas as pd
import numpy as np
from resources.config import DATA_PATH

def load_data(path = DATA_PATH):
    base_path = Path(path)
    x = pd.read_csv(base_path / "X_all.csv")
    a = pd.read_csv(base_path / "A_all.csv")
    y = pd.read_csv(base_path / "Y_all.csv")
    return x, a, y

def merged_by_sub(d1,d2):
    return pd.merge(d1,d2, on=["SUBJECT_ID"])

def drop_all_null_cols(df):
    return df.dropna(axis=1, how='all')