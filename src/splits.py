import pandas as pd

def loeo_folds(df: pd.DataFrame, id_col: str):
    esns = sorted(df[id_col].unique().tolist())
    for val_esn in esns:
        train_esns = [e for e in esns if e != val_esn]
        yield train_esns, [val_esn]
