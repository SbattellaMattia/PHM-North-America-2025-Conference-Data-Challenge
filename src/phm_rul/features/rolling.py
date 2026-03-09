"""
features/rolling.py — STEP 4d: Periodic + Rolling + Residual features.
"""
import numpy as np
import pandas as pd
from ..common import sanitize


KEY_BASE = ["temp_gradient","hptc_efficiency","thermal_stress",
            "T45_T3_ratio","fuel_per_speed",
            "corrected_core_speed","corrected_fan_speed"]
WINDOWS  = [5, 10, 30, 50, 100]


def _compute_slope(arr: np.ndarray, window: int) -> np.ndarray:
    slopes = np.zeros(len(arr))
    for i in range(len(arr)):
        seg = arr[max(0, i-window+1):i+1]
        if len(seg) > 2 and not np.isnan(seg).any():
            slopes[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0]
    return slopes


def add_periodic_and_residual_features(df: pd.DataFrame,
                                       res_cols: list) -> pd.DataFrame:
    print("\n" + "="*70)
    print("STEP 4d: PERIODIC + RESIDUAL FEATURES")
    print("="*70)

    df            = df.copy()
    res_available = [c for c in res_cols if c in df.columns]

    for esn in sorted(df["ESN"].unique()):
        mask      = df["ESN"] == esn
        df_esn    = df[mask].sort_values("Cycles").reset_index(drop=True)
        max_cycle = df_esn["Cycles"].max()
        n         = len(df_esn)

        rc = df_esn["Cycles"] / max_cycle
        df.loc[mask, "relative_cycle"]         = rc.values
        df.loc[mask, "relative_cycle_squared"] = (rc**2).values
        df.loc[mask, "relative_cycle_cubed"]   = (rc**3).values
        df.loc[mask, "cycles_since_start"]     = df_esn["Cycles"].values

        for period in [900, 1000, 1100]:
            phase = (df_esn["Cycles"] % period) / period
            df.loc[mask, f"ww_sin_{period}"] = np.sin(2*np.pi*phase).values
            df.loc[mask, f"ww_cos_{period}"] = np.cos(2*np.pi*phase).values

        for rcol in res_available:
            sig = df_esn[rcol].copy().ffill().bfill()
            sig = sig.fillna(sig.median() if pd.notna(sig.median()) else 0.0)
            arr = sig.values
            for w in WINDOWS:
                df.loc[mask, f"{rcol}_rmean_{w}"] = sig.rolling(w, min_periods=1).mean().values
                df.loc[mask, f"{rcol}_rstd_{w}"]  = sig.rolling(w, min_periods=2).std().fillna(0).values
            df.loc[mask, f"{rcol}_rate"]  = sig.diff().fillna(0).values
            df.loc[mask, f"{rcol}_accel"] = sig.diff().diff().fillna(0).values
            cs = sig.cumsum(); cm = cs.abs().max()
            df.loc[mask, f"{rcol}_cumsum_norm"] = (
                (cs/(cm+1e-6)).values if cm > 1e-6 else np.zeros(n))
            for sw in [20, 50, 100]:
                df.loc[mask, f"{rcol}_slope{sw}"] = _compute_slope(arr, sw)

        for feat in KEY_BASE:
            if feat not in df_esn.columns:
                continue
            sig = df_esn[feat].copy().ffill().bfill()
            sig = sig.fillna(sig.mean() if pd.notna(sig.mean()) else 0.0)
            for w in WINDOWS:
                df.loc[mask, f"{feat}_rmean_{w}"] = sig.rolling(w, min_periods=1).mean().values
                df.loc[mask, f"{feat}_rstd_{w}"]  = sig.rolling(w, min_periods=2).std().fillna(0).values
            df.loc[mask, f"{feat}_rate"]  = sig.diff().fillna(0).values
            df.loc[mask, f"{feat}_accel"] = sig.diff().diff().fillna(0).values
            cs = sig.cumsum(); cm = cs.abs().max()
            df.loc[mask, f"{feat}_cumsum_norm"] = (
                (cs/(cm+1e-6)).values if cm > 1e-6 else np.zeros(n))

        if "Sensed_T45_res" in df_esn.columns:
            t45r = df_esn["Sensed_T45_res"].ffill().bfill().fillna(0.0)
            if "temp_gradient" in df_esn.columns:
                df.loc[mask, "T45res_x_tempgrad"] = (
                    t45r * df_esn["temp_gradient"].ffill().bfill().fillna(0.0)).values
            if "hptc_efficiency" in df_esn.columns:
                df.loc[mask, "T45res_x_heff"] = (
                    t45r * df_esn["hptc_efficiency"].ffill().bfill().fillna(0.0)).values

        print(f"    ESN {esn}: periodic + residual features added")

    df = sanitize(df, "add_periodic_and_residual_features")
    print(f"\n  Dataset shape: {df.shape}")
    return df
