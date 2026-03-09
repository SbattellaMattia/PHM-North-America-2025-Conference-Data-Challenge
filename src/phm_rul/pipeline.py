"""
pipeline.py — STEP 6: Leave-One-Engine-Out (LOEO) loop.
Feature selection DENTRO il loop (no leakage).
"""
import numpy as np
import pandas as pd

from sklearn.impute        import SimpleImputer
from sklearn.metrics       import mean_absolute_error, r2_score
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler

from .common           import twe
from .features.selection import select_features
from .features.lagged    import create_lagged_features
from .models.ensemble    import train_full_ensemble


def run_loeo(df: pd.DataFrame, res_cols: list,
             window_size: int = 32, top_k: int = 70) -> list:
    """
    Esegue LOEO esattamente come il monolite originale:
      - feature selection su df_train ad ogni fold (no leakage)
      - ESN 104: partial train (prime 60% di cicli aggiunte al train)
    """
    esns         = sorted(df["ESN"].unique())
    fold_results = []

    print("\n" + "="*70)
    print(f"STEP 6: LOEO | window={window_size}")
    print("="*70)

    for left_out in esns:
        print(f"\n{chr(9472)*70}")
        print(f"FOLD: Leave out ESN {left_out}")
        print(f"{chr(9472)*70}")

        df_train = df[df["ESN"] != left_out].copy()
        df_test  = df[df["ESN"] == left_out].copy()

        # ── ESN 104 partial train ─────────────────────────────────────────────
        if left_out == 104:
            cutoff    = df_test["Cycles"].quantile(0.60)
            df_104_p  = df_test[df_test["Cycles"] <= cutoff].copy()
            df_train  = pd.concat([df_train, df_104_p], ignore_index=True)
            df_test   = df_test[df_test["Cycles"] > cutoff].copy()
            print("  [ESN104] Partial train: first 60% cycles added to train")

        # ── Feature selection su train (DENTRO il fold) ───────────────────────
        feature_cols = select_features(df_train, res_cols, top_k=top_k)
        print(f"  Using {len(feature_cols)} features for LOEO")

        X_train, y_train, info_train = create_lagged_features(df_train, feature_cols, window_size)
        X_test,  y_test,  info_test  = create_lagged_features(df_test,  feature_cols, window_size)

        if len(X_test) == 0:
            print(f"  ⚠️  No test samples for ESN {left_out}, skip")
            continue

        print(f"  Train: {X_train.shape}, y=[{y_train.min():.0f},{y_train.max():.0f}]")
        print(f"  Test:  {X_test.shape},  y=[{y_test.min():.0f},{y_test.max():.0f}]")

        preprocess = Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler",  StandardScaler())])
        X_tr_s = preprocess.fit_transform(X_train)
        X_te_s = preprocess.transform(X_test)

        y_pred = train_full_ensemble(X_tr_s, y_train, X_te_s, y_test,
                                     info_train=info_train, info_test=info_test)

        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)
        twe_sc = twe(y_test, y_pred)
        base   = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
        improv = (base - mae) / base * 100

        print(f"\n  ✅ RESULT ESN {left_out}:")
        print(f"     MAE:         {mae:.1f} cycles")
        print(f"     TWE score:   {twe_sc:.4f}")
        print(f"     R²:          {r2:.3f}")
        print(f"     Improvement: {improv:+.1f}%")
        print(f"     Pred range:  [{y_pred.min():.0f}, {y_pred.max():.0f}]")
        print(f"     True range:  [{y_test.min():.0f}, {y_test.max():.0f}]")

        fold_results.append({
            "left_out_esn": left_out,
            "mae":          mae,
            "twe":          twe_sc,
            "r2":           r2,
            "baseline_mae": base,
            "improvement":  improv,
            "y_true":       y_test,
            "y_pred":       y_pred,
            "info_test":    info_test,
        })

    return fold_results
