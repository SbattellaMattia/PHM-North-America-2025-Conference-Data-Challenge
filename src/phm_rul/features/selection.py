"""
features/selection.py — STEP 5: Feature selection via RF importance.
DEVE essere chiamata DENTRO il loop LOEO, solo su df_train.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute   import SimpleImputer


MUST_HAVE = [
    # Han & Liang core
    "Sensed_T45_res", "T45res_slope50", "T45res_slope20", "T45res_slope100",
    "T45res_x_tempgrad", "T45res_x_heff",
    # Cycle info
    "relative_cycle", "cycles_since_start",
    # Cumsum residui
    "Sensed_T45_res_cumsum_norm", "Sensed_T3_res_cumsum_norm",
    "Sensed_Ps3_res_cumsum_norm", "Sensed_Core_Speed_res_cumsum_norm",
    "temp_gradient_cumsum_norm",
    # MathWorks HPC-WW
    "hpc_hi_recovery", "cycles_since_ww_recovery",
    "hpc_hi_slope10",  "inter_recovery_interval",
    # Shock detector [ORIGINALE]
    "cycles_since_last_shock", "inter_shock_interval",
    "cycles_to_next_shock_est", "shock_magnitude_cumsum",
    # FFT period [ORIGINALE]
    "ww_adaptive_sin", "ww_adaptive_cos", "ww_period_est",
]

EXCLUDE = ["ESN","Cycles","Snapshot",
           "Cycles_to_WW","Cycles_to_HPC_SV","Cycles_to_HPT_SV"]


def select_features(df: pd.DataFrame, res_cols: list,
                    top_k: int = 70) -> list:
    """
    Seleziona le top_k feature per importanza RF + aggiunge MUST_HAVE.
    Chiamare SOLO su df_train (no leakage dal test ESN).
    """
    print("\n" + "="*70)
    print(f"STEP 5: FEATURE SELECTION (RF importance, top_k={top_k})")
    print("="*70)

    cands = [c for c in df.columns if c not in EXCLUDE]
    X     = SimpleImputer(strategy="median").fit_transform(df[cands].values)
    y     = df["Cycles_to_WW"].values
    ok    = ~np.isnan(y)
    X, y  = X[ok], y[ok]

    rf = RandomForestRegressor(n_estimators=150, max_depth=15,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)

    imp    = rf.feature_importances_
    ranked = np.argsort(imp)[::-1]
    top    = [cands[i] for i in ranked[:top_k]]

    print(f"  Candidates: {len(cands)} | Selected: {top_k}")
    print(f"  Top 10: {top[:10]}")

    for col in MUST_HAVE:
        if col in df.columns and col not in top:
            top.append(col)
            print(f"  Force-added: {col}")

    return top
