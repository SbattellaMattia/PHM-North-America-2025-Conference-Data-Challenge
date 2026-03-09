"""
features/lagged.py — STEP 9: Sliding-window lagged features.
"""
import numpy as np
import pandas as pd


def create_lagged_features(df: pd.DataFrame,
                           feature_cols: list,
                           window_size: int = 30):
    """
    Per ogni engine, crea X (mean/std/min/max/trend per finestra),
    y (Cycles_to_WW), info (ESN, Cycle).
    """
    X, y, info = [], [], []

    for esn in sorted(df["ESN"].unique()):
        df_e = df[df["ESN"] == esn].sort_values("Cycles").reset_index(drop=True)

        for col in feature_cols:
            if col in df_e.columns:
                s     = df_e[col].ffill().bfill()
                df_e[col] = s.fillna(s.median() if pd.notna(s.median()) else 0.0)

        for i in range(window_size, len(df_e)):
            win  = df_e.iloc[i-window_size:i]
            feat = []
            for col in feature_cols:
                v       = win[col].values.astype(float)
                all_nan = np.isnan(v).all()
                feat.append(np.nanmean(v))
                feat.append(0.0 if all_nan else np.nanstd(v))
                feat.append(0.0 if all_nan else np.nanmin(v))
                feat.append(0.0 if all_nan else np.nanmax(v))
                if not all_nan and (~np.isnan(v)).sum() > 2:
                    idx = np.where(~np.isnan(v))[0]
                    feat.append(np.polyfit(idx, v[idx], 1)[0])
                else:
                    feat.append(0.0)
            feat.extend(df_e.iloc[i][feature_cols].fillna(0.0).values)

            X.append(feat)
            y.append(df_e.iloc[i]["Cycles_to_WW"])
            info.append({"ESN": esn, "Cycle": df_e.iloc[i]["Cycles"]})

    X_arr = np.nan_to_num(np.array(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.array(y, float)
    return X_arr, y_arr, pd.DataFrame(info)
