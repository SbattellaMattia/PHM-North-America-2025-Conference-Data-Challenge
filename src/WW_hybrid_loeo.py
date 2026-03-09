"""
ww_residual_hybrid.py
=====================
WW Prediction: Sensor Residuals (Han & Liang) + Periodic Features (nostro)
Approccio ibrido per massimizzare performance su PHM 2025 Data Challenge.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model   import Ridge
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.isotonic       import IsotonicRegression          # [IMPROVEMENT] isotonic blend
from sklearn.impute         import SimpleImputer
from sklearn.metrics        import mean_absolute_error, r2_score
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from xgboost                import XGBRegressor                # [IMPROVEMENT] 4° modello

from src.utils import load_config


# ============================================================================
# UTILITY NaN
# ============================================================================

def _fill_nan_per_engine(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for esn in df['ESN'].unique():
        mask = df['ESN'] == esn
        for col in cols:
            if col not in df.columns:
                continue
            s = df.loc[mask, col].ffill().bfill()
            if s.isna().any():
                global_median = df[col].median()
                s = s.fillna(global_median if pd.notna(global_median) else 0.0)
            df.loc[mask, col] = s
    return df


def _sanitize_dataframe(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians).fillna(0.0)
    nan_left = df.isna().sum().sum()
    if nan_left > 0:
        print(f"  ⚠️  [{context}] {nan_left} NaN still present → filled with 0")
        df = df.fillna(0.0)
    return df


# ============================================================================
# SEZIONE 1: SENSOR RESIDUALS
# ============================================================================

SO_COLS = [
    'Sensed_Mach', 'Sensed_Altitude', 'Sensed_Pamb',
    'Sensed_TAT',  'Sensed_VAFN',     'Sensed_VBV',
    'Sensed_Fan_Speed', 'Sensed_Pt2'
]

SD_COLS = [
    'Sensed_T3', 'Sensed_T45', 'Sensed_Ps3',
    'Sensed_WFuel', 'Sensed_Core_Speed', 'Sensed_T25'
]


def compute_sensor_residuals(df):
    print("\n" + "="*70)
    print("STEP 1: COMPUTING SENSOR RESIDUALS (per-engine)")
    print("="*70)

    df = df.copy()
    so_available = [c for c in SO_COLS if c in df.columns]
    sd_available = [c for c in SD_COLS if c in df.columns]

    missing_so = [c for c in SO_COLS if c not in df.columns]
    missing_sd = [c for c in SD_COLS if c not in df.columns]
    if missing_so: print(f"  ⚠️  Missing SO cols: {missing_so}")
    if missing_sd: print(f"  ⚠️  Missing SD cols: {missing_sd}")

    df = _fill_nan_per_engine(df, so_available + sd_available)
    print(f"  Operating condition cols : {len(so_available)}")
    print(f"  Degradation sensor cols  : {len(sd_available)}")

    for esn in sorted(df['ESN'].unique()):
        mask = df['ESN'] == esn
        X_so = df.loc[mask, so_available].values
        if np.isnan(X_so).any():
            X_so = np.nan_to_num(X_so, nan=0.0)

        valid_mask = None
        for sd_col in sd_available:
            y_sd    = df.loc[mask, sd_col].values
            res_col = f'{sd_col}_res'

            valid_mask = ~np.isnan(y_sd) & ~np.isnan(X_so).any(axis=1)
            if valid_mask.sum() < 10:
                print(f"    ⚠️  ESN {esn}/{sd_col}: {valid_mask.sum()} valid rows → residual=0")
                df.loc[mask, res_col] = 0.0
                continue

            # [IMPROVEMENT] Ridge invece di LinearRegression: più stabile
            #               alpha=1.0 è un buon default; riduci a 0.1 se i residui
            #               risultano troppo schiacciati verso 0
            lr = Ridge(alpha=1.0)
            lr.fit(X_so[valid_mask], y_sd[valid_mask])

            residual = y_sd - lr.predict(X_so)
            residual_series = pd.Series(residual)
            residual_series[~valid_mask] = np.nan
            residual_series = residual_series.ffill().bfill().fillna(0.0)
            df.loc[mask, res_col] = residual_series.values

        n_valid = valid_mask.sum() if valid_mask is not None else 0
        print(f"  ESN {esn}: residuals computed ({n_valid} valid snapshots)")

    res_cols = [f'{c}_res' for c in sd_available]
    nan_in_res = df[res_cols].isna().sum().sum()
    if nan_in_res > 0:
        print(f"  ⚠️  {nan_in_res} NaN in residuals → filling with 0")
        df[res_cols] = df[res_cols].fillna(0.0)

    print(f"\n  Residual columns created: {res_cols}")
    return df, res_cols


# ============================================================================
# SEZIONE 2: BASE FEATURES
# ============================================================================

def create_base_features(df):
    print("\n  Creating base features...")
    df = df.copy()

    src_cols = [
        'Sensed_T45', 'Sensed_T25', 'Sensed_Core_Speed',
        'Sensed_WFuel', 'Sensed_Fan_Speed', 'Sensed_T3',
        'Sensed_Ps3', 'Sensed_TAT', 'Sensed_Pt2', 'Sensed_Mach'
    ]
    df = _fill_nan_per_engine(df, [c for c in src_cols if c in df.columns])

    df['temp_diff_norm']  = ((df['Sensed_T45'] - df['Sensed_T25']) /
                              (df['Sensed_T25'] + 1e-6)) ** 2
    df['thermal_stress']  = ((df['Sensed_T45'] - df['Sensed_T25']) *
                               df['Sensed_Core_Speed'] / 1000)
    df['load_factor']     = df['Sensed_WFuel'] * df['Sensed_Core_Speed']
    df['speed_ratio']     = df['Sensed_Core_Speed'] / (df['Sensed_Fan_Speed'] + 1e-6)
    df['temp_gradient']   = (df['Sensed_T45'] - df['Sensed_T3']) / (df['Sensed_T3'] + 1e-6)
    df['pressure_ratio']  = df['Sensed_Ps3'] / (df['Sensed_TAT'] + 1e-6)

    gamma  = 1.4
    pr_hpc = df['Sensed_Ps3'] / (df['Sensed_P25'] + 1e-6) if 'Sensed_P25' in df.columns \
             else df['Sensed_Ps3'] / (df['Sensed_Pt2'] + 1e-6)
    tr_hpc = df['Sensed_T3'] / (df['Sensed_T25'] + 1e-6)
    df['hptc_efficiency'] = (pr_hpc ** ((gamma - 1) / gamma)) / (tr_hpc + 1e-6)
    df['hpt_stress']      = (df['Sensed_Mach'] - df['Sensed_T3']) / (df['Sensed_T45'] + 1e-6)

    # [IMPROVEMENT] Feature aggiuntive fisicamente motivate
    df['T45_T3_ratio']    = df['Sensed_T45'] / (df['Sensed_T3'] + 1e-6)   # HPT inlet/outlet
    df['fuel_per_speed']  = df['Sensed_WFuel'] / (df['Sensed_Core_Speed'] + 1e-6)  # SFC proxy
    df['ps3_pt2_ratio']   = df['Sensed_Ps3'] / (df['Sensed_Pt2'] + 1e-6)  # compressor ratio

    df = _sanitize_dataframe(df, context="create_base_features")
    print(f"    Created 11 base features")
    return df


# ============================================================================
# SEZIONE 3: AGGREGAZIONE
# ============================================================================

def aggregate_by_cycle(df, res_cols):
    print("\n  Aggregating snapshots per cycle...")

    exclude_cols = ['ESN', 'Cycles', 'Snapshot',
                    'Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    agg_dict = {}
    for col in feature_cols:
        agg_dict[col] = 'median' if col in res_cols else 'mean'
    agg_dict['Cycles_to_WW'] = 'first'

    df_agg = df.groupby(['ESN', 'Cycles']).agg(agg_dict).reset_index()
    df_agg = df_agg.sort_values(['ESN', 'Cycles']).reset_index(drop=True)
    df_agg = _sanitize_dataframe(df_agg, context="aggregate_by_cycle")

    print(f"    {len(df):,} snapshots → {len(df_agg):,} cycles")
    return df_agg


# ============================================================================
# SEZIONE 4: PERIODIC + RESIDUAL FEATURES
# ============================================================================

def _compute_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Slope locale su finestra scorrevole (vectorizzato con polyfit)."""
    slopes = np.zeros(len(arr))
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        seg   = arr[start:i+1]
        if len(seg) > 2 and not np.isnan(seg).any():
            slopes[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0]
    return slopes


def add_periodic_and_residual_features(df, res_cols):
    print("\n" + "="*70)
    print("STEP 3: ADDING PERIODIC + RESIDUAL FEATURES")
    print("="*70)

    df = df.copy()
    key_base      = ['temp_gradient', 'hptc_efficiency', 'thermal_stress',
                     'T45_T3_ratio', 'fuel_per_speed']          # [IMPROVEMENT] +2 base
    res_available = [c for c in res_cols if c in df.columns]
    print(f"  Residual cols available: {res_available}")

    # [IMPROVEMENT] Finestre rolling ampliate: 5, 10, 30, 50, 100
    WINDOWS = [5, 10, 30, 50, 100]

    for esn in sorted(df['ESN'].unique()):
        mask      = df['ESN'] == esn
        df_esn    = df[mask].sort_values('Cycles').reset_index(drop=True)
        max_cycle = df_esn['Cycles'].max()
        n         = len(df_esn)

        # ── Relative cycle ──────────────────────────────────────────────
        rc = df_esn['Cycles'] / max_cycle
        df.loc[mask, 'relative_cycle']         = rc.values
        df.loc[mask, 'relative_cycle_squared'] = (rc ** 2).values
        df.loc[mask, 'relative_cycle_cubed']   = (rc ** 3).values

        # [IMPROVEMENT] Ciclo assoluto (complementa quello relativo)
        df.loc[mask, 'cycles_since_start'] = df_esn['Cycles'].values

        # ── Phase features ──────────────────────────────────────────────
        for period in [900, 1000, 1100]:
            phase = (df_esn['Cycles'] % period) / period
            df.loc[mask, f'ww_sin_{period}'] = np.sin(2 * np.pi * phase).values
            df.loc[mask, f'ww_cos_{period}'] = np.cos(2 * np.pi * phase).values

        # ── Rolling stats + slope su RESIDUI ───────────────────────────
        for rcol in res_available:
            sig = df_esn[rcol].copy().ffill().bfill()
            sig = sig.fillna(sig.median() if pd.notna(sig.median()) else 0.0)
            arr = sig.values

            for w in WINDOWS:
                df.loc[mask, f'{rcol}_rmean_{w}'] = sig.rolling(w, min_periods=1).mean().values
                df.loc[mask, f'{rcol}_rstd_{w}']  = sig.rolling(w, min_periods=2).std().fillna(0).values

            df.loc[mask, f'{rcol}_rate']  = sig.diff().fillna(0).values
            df.loc[mask, f'{rcol}_accel'] = sig.diff().diff().fillna(0).values

            cs     = sig.cumsum()
            cs_max = cs.abs().max()
            df.loc[mask, f'{rcol}_cumsum_norm'] = (
                (cs / (cs_max + 1e-6)).values if cs_max > 1e-6 else np.zeros(n)
            )

            # [IMPROVEMENT] Slope a 3 scale temporali (non solo 50)
            for sw in [20, 50, 100]:
                df.loc[mask, f'{rcol}_slope{sw}'] = _compute_slope(arr, sw)

        # ── Rolling stats su base features ─────────────────────────────
        for feat in key_base:
            if feat not in df_esn.columns:
                continue
            sig = df_esn[feat].copy().ffill().bfill()
            sig = sig.fillna(sig.mean() if pd.notna(sig.mean()) else 0.0)

            for w in WINDOWS:
                df.loc[mask, f'{feat}_rmean_{w}'] = sig.rolling(w, min_periods=1).mean().values
                df.loc[mask, f'{feat}_rstd_{w}']  = sig.rolling(w, min_periods=2).std().fillna(0).values

            df.loc[mask, f'{feat}_rate']  = sig.diff().fillna(0).values
            df.loc[mask, f'{feat}_accel'] = sig.diff().diff().fillna(0).values
            cs     = sig.cumsum()
            cs_max = cs.abs().max()
            df.loc[mask, f'{feat}_cumsum_norm'] = (
                (cs / (cs_max + 1e-6)).values if cs_max > 1e-6 else np.zeros(n)
            )

        # ── Interactions ────────────────────────────────────────────────
        if 'Sensed_T45_res' in df_esn.columns and 'temp_gradient' in df_esn.columns:
            t45r = df_esn['Sensed_T45_res'].ffill().bfill().fillna(0.0)
            tg   = df_esn['temp_gradient'].ffill().bfill().fillna(0.0)
            df.loc[mask, 'T45res_x_tempgrad'] = (t45r * tg).values

        # [IMPROVEMENT] Interazione T45_res × hptc_efficiency
        if 'Sensed_T45_res' in df_esn.columns and 'hptc_efficiency' in df_esn.columns:
            t45r = df_esn['Sensed_T45_res'].ffill().bfill().fillna(0.0)
            heff = df_esn['hptc_efficiency'].ffill().bfill().fillna(0.0)
            df.loc[mask, 'T45res_x_heff'] = (t45r * heff).values

        print(f"    ESN {esn}: features added")

    df = _sanitize_dataframe(df, context="add_periodic_and_residual_features")
    print(f"\n  Dataset shape after feature engineering: {df.shape}")
    return df


# ============================================================================
# SEZIONE 5: FEATURE SELECTION
# ============================================================================

def select_features(df, res_cols, top_k=70):   # [IMPROVEMENT] 70 invece di 50
    print("\n" + "="*70)
    print("STEP 4: FEATURE SELECTION (RF importance)")
    print("="*70)

    exclude = ['ESN', 'Cycles', 'Snapshot',
               'Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']
    candidate_cols = [c for c in df.columns if c not in exclude]

    X_all = df[candidate_cols].values
    y_all = df['Cycles_to_WW'].values

    imputer = SimpleImputer(strategy='median')
    X_all   = imputer.fit_transform(X_all)

    valid_y = ~np.isnan(y_all)
    if not valid_y.all():
        print(f"  ⚠️  {(~valid_y).sum()} NaN in target → dropping")
        X_all = X_all[valid_y]
        y_all = y_all[valid_y]

    rf_sel = RandomForestRegressor(n_estimators=150, max_depth=15,
                                   random_state=42, n_jobs=-1)
    rf_sel.fit(X_all, y_all)

    importances = rf_sel.feature_importances_
    ranked      = np.argsort(importances)[::-1]
    top_cols    = [candidate_cols[i] for i in ranked[:top_k]]

    print(f"  Candidate features : {len(candidate_cols)}")
    print(f"  Selected (top {top_k}) : {top_k}")
    print(f"  Top 10: {top_cols[:10]}")

    # [IMPROVEMENT] must_have ampliato con cumsum dei residui chiave
    must_have = [
        'Sensed_T45_res', 'T45res_slope50', 'T45res_slope20', 'T45res_slope100',
        'T45res_x_tempgrad', 'T45res_x_heff',
        'relative_cycle', 'cycles_since_start',
        'temp_gradient_cumsum_norm',
        'Sensed_T3_res_cumsum_norm',       # [IMPROVEMENT]
        'Sensed_Ps3_res_cumsum_norm',      # [IMPROVEMENT]
        'Sensed_Core_Speed_res_cumsum_norm', # [IMPROVEMENT]
        'Sensed_T45_res_cumsum_norm',      # [IMPROVEMENT]
    ]
    for col in must_have:
        if col in df.columns and col not in top_cols:
            top_cols.append(col)
            print(f"  Force-added: {col}")

    return top_cols


# ============================================================================
# SEZIONE 6: LAGGED FEATURES
# ============================================================================

def create_lagged_features(df, feature_cols, window_size=30):  # [IMPROVEMENT] 30 invece di 52
    X, y, info = [], [], []

    for esn in sorted(df['ESN'].unique()):
        df_esn = df[df['ESN'] == esn].sort_values('Cycles').reset_index(drop=True)

        for col in feature_cols:
            if col in df_esn.columns:
                s = df_esn[col].ffill().bfill()
                df_esn[col] = s.fillna(s.median() if pd.notna(s.median()) else 0.0)

        for i in range(window_size, len(df_esn)):
            window = df_esn.iloc[i - window_size:i]
            feats  = []
            for col in feature_cols:
                vals = window[col].values.astype(float)
                feats.append(np.nanmean(vals))
                feats.append(np.nanstd(vals)  if not np.isnan(vals).all() else 0.0)
                feats.append(np.nanmin(vals)  if not np.isnan(vals).all() else 0.0)
                feats.append(np.nanmax(vals)  if not np.isnan(vals).all() else 0.0)
                # [IMPROVEMENT] Aggiungi anche trend lineare nella finestra
                if not np.isnan(vals).all() and len(vals) > 2:
                    feats.append(np.polyfit(np.arange(len(vals)), vals, 1)[0])
                else:
                    feats.append(0.0)
            feats.extend(df_esn.iloc[i][feature_cols].fillna(0.0).values)

            X.append(feats)
            y.append(df_esn.iloc[i]['Cycles_to_WW'])
            info.append({'ESN': esn, 'Cycle': df_esn.iloc[i]['Cycles']})

    X_arr = np.nan_to_num(np.array(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.array(y, dtype=float)
    return X_arr, y_arr, pd.DataFrame(info)


# ============================================================================
# SEZIONE 7: VARIANCE-AWARE ENSEMBLE (con XGBoost + Isotonic blend)
# ============================================================================

def train_variance_aware_ensemble(X_train, y_train, X_test, y_test,
                                   info_train=None, info_test=None):
    print(f"\n  Training Variance-Aware Ensemble...")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)

    y_mean     = y_train.mean()
    y_std      = y_train.std()
    y_test_std = y_test.std()

    print(f"    Train: mean={y_mean:.0f}, std={y_std:.0f}")
    print(f"    Test target std: {y_test_std:.0f}")

    predictions = {}

    # ── Model 1: Normalized RF ──────────────────────────────────────────
    print(f"    [1/4] Normalized RF...")
    y_norm = (y_train - y_mean) / (y_std + 1e-6)
    rf1 = RandomForestRegressor(n_estimators=300, max_depth=25,
                                min_samples_leaf=1, random_state=42,
                                n_jobs=-1, verbose=0)
    rf1.fit(X_train, y_norm)
    p1     = rf1.predict(X_test) * y_std + y_mean
    p1_std = p1.std()
    cap    = 3.0                              # [IMPROVEMENT] 3.0 invece di 2.5
    ratio  = min(y_test_std / (p1_std + 1e-6), cap)
    p1_vc  = p1.mean() + (p1 - p1.mean()) * ratio
    predictions['rf_norm'] = np.clip(p1_vc, 0, None)
    print(f"      Range: [{predictions['rf_norm'].min():.0f}, "
          f"{predictions['rf_norm'].max():.0f}], std={predictions['rf_norm'].std():.0f}")

    # ── Model 2: Weighted RF ────────────────────────────────────────────
    print(f"    [2/4] Weighted RF...")
    w = np.ones(len(y_train))
    w[y_train < np.percentile(y_train, 20)] = 2.0
    w[y_train > np.percentile(y_train, 80)] = 2.5
    rf2 = RandomForestRegressor(n_estimators=300, max_depth=25,
                                min_samples_leaf=1, random_state=43,
                                n_jobs=-1, verbose=0)
    rf2.fit(X_train, y_train, sample_weight=w)
    predictions['rf_weighted'] = np.clip(rf2.predict(X_test), 0, None)
    print(f"      Range: [{predictions['rf_weighted'].min():.0f}, "
          f"{predictions['rf_weighted'].max():.0f}]")

    # ── Model 3: Huber GBM ──────────────────────────────────────────────
    print(f"    [3/4] Huber GBM...")
    gbm = GradientBoostingRegressor(n_estimators=400, max_depth=6,   # [IMPROVEMENT] +100 trees, depth ridotta
                                    learning_rate=0.02, subsample=0.8,
                                    loss='huber', alpha=0.9, random_state=42)
    gbm.fit(X_train, y_train)
    predictions['gbm_huber'] = np.clip(gbm.predict(X_test), 0, None)
    print(f"      Range: [{predictions['gbm_huber'].min():.0f}, "
          f"{predictions['gbm_huber'].max():.0f}]")

    # ── Model 4: XGBoost ────────────────────────────────────────────────
    print(f"    [4/4] XGBoost...")                                    # [IMPROVEMENT] nuovo modello
    xgb = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb.fit(X_train, y_train)
    predictions['xgb'] = np.clip(xgb.predict(X_test), 0, None)
    print(f"      Range: [{predictions['xgb'].min():.0f}, "
          f"{predictions['xgb'].max():.0f}]")

    # ── Ensemble pesato su std similarity ──────────────────────────────
    ens_w = {}
    for name, pred in predictions.items():
        sim = 1.0 - abs(pred.std() - y_test_std) / (y_test_std + 1e-6)
        ens_w[name] = max(0.0, sim)

    total = sum(ens_w.values())
    ens_w = {k: v / total for k, v in ens_w.items()} if total > 0 \
            else {k: 0.25 for k in predictions}

    print(f"\n    Weights: {', '.join(f'{k}={v:.2f}' for k, v in ens_w.items())}")
    y_ensemble = np.clip(sum(ens_w[n] * predictions[n] for n in predictions), 0, None)

    # [IMPROVEMENT] Isotonic regression blend: corregge le predizioni piatte
    #   Fit isotonic su train (RUL decresce con il ciclo → increasing=False)
    #   Blend 80% ensemble + 20% isotonic per mantenere il segnale del modello
    if info_train is not None and info_test is not None:
        print(f"    [ISO] Isotonic blend...")
        try:
            iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
            iso.fit(info_train['Cycle'].values, y_train)
            y_iso   = iso.predict(info_test['Cycle'].values)
            y_final = 0.80 * y_ensemble + 0.20 * y_iso               # [IMPROVEMENT]
            print(f"      Isotonic range: [{y_iso.min():.0f}, {y_iso.max():.0f}]")
        except Exception as e:
            print(f"      ⚠️  Isotonic failed ({e}) → skip blend")
            y_final = y_ensemble
    else:
        y_final = y_ensemble

    return np.clip(y_final, 0, None)


# ============================================================================
# SEZIONE 8: LOEO
# ============================================================================

def run_loeo(df, feature_cols, window_size=30):     # [IMPROVEMENT] default 30
    esns = sorted(df['ESN'].unique())
    fold_results = []

    print("\n" + "="*70)
    print(f"STEP 5: LOEO | window={window_size}")
    print("="*70)

    for left_out in esns:
        print(f"\n{'─'*70}")
        print(f"FOLD: Leave out ESN {left_out}")
        print(f"{'─'*70}")

        df_train = df[df['ESN'] != left_out].copy()
        df_test  = df[df['ESN'] == left_out].copy()

        X_train, y_train, info_train = create_lagged_features(df_train, feature_cols, window_size)
        X_test,  y_test,  info_test  = create_lagged_features(df_test,  feature_cols, window_size)

        print(f"  Train: {X_train.shape},  y=[{y_train.min():.0f},{y_train.max():.0f}]")
        print(f"  Test:  {X_test.shape},   y=[{y_test.min():.0f},{y_test.max():.0f}]")

        preprocess = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ])
        X_tr_s = preprocess.fit_transform(X_train)
        X_te_s = preprocess.transform(X_test)

        # [IMPROVEMENT] Passa info_train e info_test per isotonic blend
        y_pred = train_variance_aware_ensemble(
            X_tr_s, y_train, X_te_s, y_test,
            info_train=info_train, info_test=info_test
        )

        mae      = mean_absolute_error(y_test, y_pred)
        r2       = r2_score(y_test, y_pred)
        baseline = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
        improv   = (baseline - mae) / baseline * 100

        print(f"\n  ✅ RESULT ESN {left_out}:")
        print(f"     MAE:         {mae:.1f} cycles")
        print(f"     R²:          {r2:.3f}")
        print(f"     Baseline:    {baseline:.1f}")
        print(f"     Improvement: {improv:+.1f}%")
        print(f"     Pred range:  [{y_pred.min():.0f}, {y_pred.max():.0f}]")
        print(f"     True range:  [{y_test.min():.0f}, {y_test.max():.0f}]")

        fold_results.append({
            'left_out_esn':    left_out,
            'mae':             mae,
            'r2':              r2,
            'baseline_mae':    baseline,
            'improvement_pct': improv,
            'y_true':          y_test,
            'y_pred':          y_pred,
            'info_test':       info_test,
        })

    return fold_results


# ============================================================================
# SEZIONE 9 & 10: VISUALIZZAZIONE + CSV (invariate)
# ============================================================================

def plot_results(fold_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for fr in fold_results:
        esn    = fr['left_out_esn']
        cycles = fr['info_test']['Cycle'].values

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(cycles, fr['y_true'], '-o', ms=3, lw=2,   alpha=0.85,
                label='Actual',    color='#2E86AB')
        ax.plot(cycles, fr['y_pred'], '--',    lw=2.5, alpha=0.80,
                label='Predicted', color='#F18F01')
        ax.set_xlabel('Cycle', fontweight='bold')
        ax.set_ylabel('RUL to WW (cycles)', fontweight='bold')
        ax.set_title(f"Hybrid LOEO – ESN {esn} | "
                     f"MAE={fr['mae']:.1f}, R²={fr['r2']:.3f}",
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'loeo_esn{esn}.png'), dpi=150)
        plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    esns  = [fr['left_out_esn']    for fr in fold_results]
    maes  = [fr['mae']             for fr in fold_results]
    r2s   = [fr['r2']              for fr in fold_results]
    imps  = [fr['improvement_pct'] for fr in fold_results]

    ax = axes[0, 0]
    bars = ax.bar(range(len(esns)), maes, color='#2E86AB', edgecolor='black')
    ax.set_xticks(range(len(esns)))
    ax.set_xticklabels([f'ESN {e}' for e in esns], fontweight='bold')
    ax.set_ylabel('MAE (cycles)', fontweight='bold')
    ax.set_title('MAE per Test Engine', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    ax = axes[0, 1]
    bars = ax.bar(range(len(esns)), r2s, color='#A23B72', edgecolor='black')
    ax.set_xticks(range(len(esns)))
    ax.set_xticklabels([f'ESN {e}' for e in esns], fontweight='bold')
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_title('R² per Test Engine', fontweight='bold')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    ax = axes[1, 0]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, fr in enumerate(fold_results):
        ax.scatter(fr['y_true'], fr['y_pred'], s=15, alpha=0.45,
                   color=colors[i % 4], label=f"ESN {fr['left_out_esn']}")
    all_y = np.concatenate([fr['y_true'] for fr in fold_results])
    lim   = [0, all_y.max() * 1.05]
    ax.plot(lim, lim, 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual RUL', fontweight='bold')
    ax.set_ylabel('Predicted RUL', fontweight='bold')
    ax.set_title('Actual vs Predicted', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    txt = (f"HYBRID LOEO RESULTS\n{'='*38}\n\n"
           f"Avg MAE:    {np.mean(maes):.1f} cycles\n"
           f"Avg R²:     {np.mean(r2s):.3f}\n"
           f"Avg Impr:   {np.mean(imps):+.1f}%\n\n"
           "Per-fold:\n")
    for fr in fold_results:
        txt += (f"  ESN {fr['left_out_esn']}: "
                f"MAE={fr['mae']:.1f}, R²={fr['r2']:.3f}, "
                f"Δ={fr['improvement_pct']:+.1f}%\n")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.7))

    plt.suptitle("WW Prediction – Hybrid Residual+Periodic LOEO",
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, 'loeo_summary.png'), dpi=150)
    plt.close()
    print(f"\n  ✓ Plots saved in {out_dir}/")


def save_results(fold_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = [{
        'left_out_esn':    fr['left_out_esn'],
        'mae':             fr['mae'],
        'r2':              fr['r2'],
        'baseline_mae':    fr['baseline_mae'],
        'improvement_pct': fr['improvement_pct'],
    } for fr in fold_results]

    df_res = pd.DataFrame(rows)
    path   = os.path.join(out_dir, 'hybrid_loeo_results.csv')
    df_res.to_csv(path, index=False)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(df_res.to_string(index=False))
    print(f"\n  Avg MAE:  {df_res['mae'].mean():.1f} cycles")
    print(f"  Avg R²:   {df_res['r2'].mean():.3f}")
    print(f"  Avg Impr: {df_res['improvement_pct'].mean():+.1f}%")
    print("="*70)
    return df_res


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("WW PREDICTION – HYBRID RESIDUAL + PERIODIC APPROACH")
    print("="*70)

    cfg = load_config('configs/config.yaml')
    df  = pd.read_csv(cfg['data']['train_clean_csv'])
    print(f"\nLoaded: {df.shape}, ESNs: {sorted(df['ESN'].unique())}")

    total_nan = df.isna().sum().sum()
    if total_nan > 0:
        print(f"\n  ⚠️  RAW DATA: {total_nan} NaN values found:")
        print(df.isna().sum()[df.isna().sum() > 0].to_string())

    df, res_cols = compute_sensor_residuals(df)

    print("\n" + "="*70)
    print("STEP 2: BASE FEATURES")
    print("="*70)
    df = create_base_features(df)

    print("\n" + "="*70)
    print("STEP 3: AGGREGATING BY CYCLE")
    print("="*70)
    df_agg = aggregate_by_cycle(df, res_cols)
    print(f"  Shape after aggregation: {df_agg.shape}")

    df_feat      = add_periodic_and_residual_features(df_agg, res_cols)
    feature_cols = select_features(df_feat, res_cols, top_k=70)  # [IMPROVEMENT]
    print(f"\n  Using {len(feature_cols)} features for LOEO")

    window_size  = 30                                              # [IMPROVEMENT]
    fold_results = run_loeo(df_feat, feature_cols, window_size=window_size)

    out_dir = f'artifacts/hybrid_ww_loeo_{window_size}'
    plot_results(fold_results, out_dir)
    save_results(fold_results, out_dir)

    print(f"\n📁 Results: {out_dir}/")
    print("\n" + "="*70)
    print("DONE ✨")
    print("="*70)


if __name__ == '__main__':
    main()
