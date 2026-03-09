"""
features/residuals.py — STEP 1: Sensor residuals (Han & Liang).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from ..common import fill_nan_per_engine

SO_COLS = ["Sensed_Mach", "Sensed_Altitude", "Sensed_Pamb",
           "Sensed_TAT",  "Sensed_VAFN",     "Sensed_VBV",
           "Sensed_Fan_Speed", "Sensed_Pt2"]

SD_COLS = ["Sensed_T3", "Sensed_T45", "Sensed_Ps3",
           "Sensed_WFuel", "Sensed_Core_Speed", "Sensed_T25"]


def compute_sensor_residuals(df: pd.DataFrame):
    print("\n" + "="*70)
    print("STEP 1: SENSOR RESIDUALS (per-engine, Ridge regression)")
    print("="*70)

    df       = df.copy()
    so_avail = [c for c in SO_COLS if c in df.columns]
    sd_avail = [c for c in SD_COLS if c in df.columns]

    missing = [c for c in SO_COLS + SD_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠️  Missing cols: {missing}")

    df = fill_nan_per_engine(df, so_avail + sd_avail)

    for esn in sorted(df["ESN"].unique()):
        mask = df["ESN"] == esn
        X_so = np.nan_to_num(df.loc[mask, so_avail].values, nan=0.0)

        valid_mask = None
        for sd_col in sd_avail:
            y_sd    = df.loc[mask, sd_col].values
            res_col = f"{sd_col}_res"
            valid_mask = ~np.isnan(y_sd) & ~np.isnan(X_so).any(axis=1)
            if valid_mask.sum() < 10:
                df.loc[mask, res_col] = 0.0
                continue
            lr  = Ridge(alpha=1.0)
            lr.fit(X_so[valid_mask], y_sd[valid_mask])
            res = pd.Series(y_sd - lr.predict(X_so))
            res[~valid_mask] = np.nan
            res = res.ffill().bfill().fillna(0.0)
            df.loc[mask, res_col] = res.values

        n_valid = valid_mask.sum() if valid_mask is not None else 0
        print(f"  ESN {esn}: {n_valid} valid snapshots")

    res_cols = [f"{c}_res" for c in sd_avail]
    nan_check = df[res_cols].isna().sum().sum()
    if nan_check:
        print(f"  ⚠️  {nan_check} NaN in residuals → 0")
        df[res_cols] = df[res_cols].fillna(0.0)

    print(f"  Residual cols: {res_cols}")
    return df, res_cols
