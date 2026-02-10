import pandas as pd
import numpy as np

def add_temporal_features(wide_df, id_col, cycle_col, sensor_cols, windows=[50, 100]):
    """
    Aggiungi rolling stats e delta per sensori nel wide table.
    Input: wide_df con (id_col, cycle_col, sensor×snapshot columns)
    """
    df = wide_df.copy()
    df = df.sort_values([id_col, cycle_col]).reset_index(drop=True)
    
    for s in sensor_cols:
        # delta rispetto a ciclo precedente
        df[f"{s}_delta"] = df.groupby(id_col)[s].diff()
        
        # rolling mean/std
        for w in windows:
            df[f"{s}_rolling_mean_{w}"] = df.groupby(id_col)[s].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{s}_rolling_std_{w}"] = df.groupby(id_col)[s].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
    
    # fillna per delta (primo ciclo di ogni ESN)
    df = df.fillna(0.0)
    return df
