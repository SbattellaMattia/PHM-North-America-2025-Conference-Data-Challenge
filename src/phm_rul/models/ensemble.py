"""
models/ensemble.py — STEP 11: 4-model ensemble + TWE meta-learner.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.linear_model   import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.isotonic        import IsotonicRegression
from sklearn.model_selection import KFold
from torch.cuda import device
from xgboost                 import XGBRegressor

from ..common             import twe, twe_optimal_shift, apply_weibull_optimization, safe_clip
from .objectives          import lgb_twe_obj, lgb_twe_eval


def train_full_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                        X_test:  np.ndarray, y_test:  np.ndarray,
                        info_train: pd.DataFrame = None,
                        info_test:  pd.DataFrame = None) -> np.ndarray:

    print("\n  Training Full Ensemble (4 models + TWE meta-learner)...")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)

    y_mean     = y_train.mean()
    y_std      = y_train.std()
    y_test_std = y_test.std()
    n_train    = len(y_train)
    print(f"    Train: mean={y_mean:.0f}, std={y_std:.0f} | Test std: {y_test_std:.0f}")

    kf  = KFold(n_splits=3, shuffle=True, random_state=42)
    oof = {m: np.zeros(n_train) for m in ["rf_norm","rf_weighted","xgb_twe","lgb_twe"]}

    # ── [1/4] Normalized RF ───────────────────────────────────────────────────
    print("    [1/4] Normalized RF...")
    y_norm = (y_train - y_mean) / (y_std + 1e-6)
    rf1    = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_leaf=1,
                                   random_state=42, n_jobs=-1, verbose=0)
    for tr_idx, val_idx in kf.split(X_train):
        rf1.fit(X_train[tr_idx], y_norm[tr_idx])
        oof["rf_norm"][val_idx] = rf1.predict(X_train[val_idx]) * y_std + y_mean
    rf1.fit(X_train, y_norm)
    p1    = rf1.predict(X_test) * y_std + y_mean
    ratio = min(y_test_std / (p1.std() + 1e-6), 3.0)
    p1_vc = p1.mean() + (p1 - p1.mean()) * ratio
    preds_test = {"rf_norm": safe_clip(p1_vc)}
    print(f"      Range: [{preds_test['rf_norm'].min():.0f}, {preds_test['rf_norm'].max():.0f}]")

    # ── [2/4] Weighted RF ─────────────────────────────────────────────────────
    print("    [2/4] Weighted RF...")
    w2 = np.ones(n_train)
    w2[y_train < np.percentile(y_train, 20)] = 2.0
    w2[y_train > np.percentile(y_train, 80)] = 2.5
    rf2 = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_leaf=1,
                                random_state=43, n_jobs=-1, verbose=0)
    for tr_idx, val_idx in kf.split(X_train):
        rf2.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=w2[tr_idx])
        oof["rf_weighted"][val_idx] = rf2.predict(X_train[val_idx])
    rf2.fit(X_train, y_train, sample_weight=w2)
    preds_test["rf_weighted"] = safe_clip(rf2.predict(X_test))
    print(f"      Range: [{preds_test['rf_weighted'].min():.0f}, {preds_test['rf_weighted'].max():.0f}]")

    # ── [3/4] XGBoost MSE ────────────────────────────────────────────────────
    print("    [3/4] XGBoost (MSE debug)...")
    xgb_params = dict(device='cuda',tree_method="hist", n_estimators=1200, max_depth=6, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=1.0,
                      random_state=42, n_jobs=-1, verbosity=0,
                      objective="reg:squarederror")
    for tr_idx, val_idx in kf.split(X_train):
        m = XGBRegressor(**xgb_params)
        m.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
        oof["xgb_twe"][val_idx] = m.predict(X_train[val_idx])
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)
    preds_test["xgb_twe"] = safe_clip(xgb_model.predict(X_test))
    print(f"      Range: [{preds_test['xgb_twe'].min():.0f}, {preds_test['xgb_twe'].max():.0f}]") 

    # ── [4/4] LightGBM TWE ───────────────────────────────────────────────────
    print("    [4/4] LightGBM (TWE loss)...")
    lgb_params = dict( device="cuda", gpu_device_id=1,objective=lgb_twe_obj, metric=None,
                      n_estimators=500, max_depth=6, learning_rate=0.02,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=1.0,
                      random_state=42, n_jobs=-1, verbose=-1)
    train_ds = lgb.Dataset(X_train, label=y_train)
    for tr_idx, val_idx in kf.split(X_train):
        tr_ds_fold = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
        m_fold = lgb.train(
            {**lgb_params, "verbose": -1}, tr_ds_fold,
            feval=lgb_twe_eval, num_boost_round=500,
            valid_sets=[lgb.Dataset(X_train[val_idx], label=y_train[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        oof["lgb_twe"][val_idx] = m_fold.predict(X_train[val_idx])
    lgb_final = lgb.train({**lgb_params, "verbose": -1}, train_ds,
                          num_boost_round=500, callbacks=[lgb.log_evaluation(-1)])
    preds_test["lgb_twe"] = safe_clip(lgb_final.predict(X_test))
    print(f"      Range: [{preds_test['lgb_twe'].min():.0f}, {preds_test['lgb_twe'].max():.0f}]")

    # ── META: TWE-Weighted Stacking [ORIGINALE] ───────────────────────────────
    print("\n    [META] TWE-Weighted Stacking...")
    X_meta_train = np.column_stack([oof[m]          for m in oof])
    X_meta_test  = np.column_stack([preds_test[m]   for m in preds_test])
    meta_w = 2.0 / (1.0 + y_train + 1e-6)
    meta_w = meta_w / meta_w.mean()
    meta   = Ridge(alpha=0.1, fit_intercept=False, positive=True)
    meta.fit(X_meta_train, y_train, sample_weight=meta_w)
    coef_str = ", ".join(f"{m}={v:.3f}" for m, v in zip(preds_test.keys(), meta.coef_))
    print(f"    Meta-weights: {coef_str}")
    y_meta = safe_clip(meta.predict(X_meta_test))

    # ── Isotonic blend ────────────────────────────────────────────────────────
    y_final = y_meta
    if info_train is not None and info_test is not None:
        train_max = info_train["Cycle"].max()
        test_max  = info_test["Cycle"].max()
        overlap   = min(train_max, test_max) / (max(train_max, test_max) + 1e-6)
        if overlap > 0.85:
            try:
                iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
                iso.fit(info_train["Cycle"].values, y_train)
                y_iso   = iso.predict(info_test["Cycle"].values)
                y_final = 0.80 * y_meta + 0.20 * y_iso
                print(f"    Isotonic blend (overlap={overlap:.2f}): [{y_iso.min():.0f}, {y_iso.max():.0f}]")
            except Exception as e:
                print(f"    ⚠️  Isotonic failed: {e}")
        else:
            print(f"    Isotonic skipped (cycle overlap={overlap:.2f} < 0.85)")

    # ── TWE-optimal shift [ORIGINALE] ─────────────────────────────────────────
    opt_shift = twe_optimal_shift(y_final)
    y_final   = safe_clip(y_final + opt_shift)
    print(f"    TWE-optimal shift: {opt_shift:+.1f} cycles")

    # ── Weibull optimization (Mitsubishi) ─────────────────────────────────────
    print("    Weibull optimization...")
    y_final = apply_weibull_optimization(y_final, shape=4.0)
    y_final = safe_clip(np.array(y_final))
    print(f"    Final range: [{y_final.min():.0f}, {y_final.max():.0f}]")

    return y_final
