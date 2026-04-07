"""
ww_residual_hybrid_v3.py
========================
WW Prediction: Hybrid Residual + Periodic + Original Contributions
PHM 2025 Data Challenge

NOVITÀ v3:
  - Custom TWE loss (XGBoost + LightGBM)
  - HPC-WW Recovery Feature (MathWorks)
  - Adaptive WW Period Detection via FFT [ORIGINALE]
  - Residual Shock Detector [ORIGINALE]
  - TWE-Aware Variance Correction [ORIGINALE]
  - TWE-Weighted Stacking Meta-Learner [ORIGINALE]
  - Weibull Optimization (Mitsubishi)
  - Gestione NaN robusta ovunque
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal          import periodogram
from scipy.optimize        import minimize_scalar
from scipy.stats           import weibull_min

from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute        import SimpleImputer
from sklearn.metrics       import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import KFold

from xgboost               import XGBRegressor
import lightgbm            as lgb

from src.utils import load_config


# ============================================================================
# UTILITY: NaN + TWE
# ============================================================================

def _fill_nan_per_engine(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for esn in df['ESN'].unique():
        mask = df['ESN'] == esn
        for col in cols:
            if col not in df.columns:
                continue
            s = df.loc[mask, col].ffill().bfill()
            if s.isna().any():
                gm = df[col].median()
                s  = s.fillna(gm if pd.notna(gm) else 0.0)
            df.loc[mask, col] = s
    return df


def _sanitize(df: pd.DataFrame, ctx: str = "") -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].median()).fillna(0.0)
    left = df.isna().sum().sum()
    if left:
        print(f"  ⚠️  [{ctx}] {left} NaN → 0")
        df = df.fillna(0.0)
    return df

def twe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TWE ufficiale PHM 2025 — alpha=0.01, beta=1/max(y_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    alpha  = 0.01
    beta   = 1.0 / (y_true.max() + 1e-6)
    diff   = y_pred - y_true
    w      = np.where(diff >= 0,
                      2.0 * alpha / (1.0 + beta * y_true),
                      1.0 * alpha / (1.0 + beta * y_true))
    return float(np.mean(w * diff**2))

def twe_optimal_shift(y_pred: np.ndarray,
                      search_range: float = 400,
                      n_points: int = 300) -> float:
    """
    [ORIGINALE] Trova il bias additivo che minimizza il TWE atteso.
    Usa y_pred come proxy della distribuzione vera.
    """
    shifts = np.linspace(-search_range, search_range, n_points)
    best_shift, best_val = 0.0, np.inf
    for s in shifts:
        yp  = np.clip(y_pred + s, 0, None)
        val = twe(y_pred, yp)        # confronto interno come proxy
        if val < best_val:
            best_val, best_shift = val, s
    return best_shift

def twe_optimal_stretch(y_pred: np.ndarray,
                        k_range: tuple = (0.8, 2.0),
                        n_points: int = 50) -> float:
    """
    Trova il fattore moltiplicativo k che minimizza il TWE atteso.
    y_corrected = mean + (y_pred - mean) * k
    """
    mu = y_pred.mean()
    ks = np.linspace(k_range[0], k_range[1], n_points)
    best_k, best_val = 1.0, np.inf
    for k in ks:
        yp  = np.clip(mu + (y_pred - mu) * k, 0, None)
        val = twe(y_pred, yp)   # proxy interno
        if val < best_val:
            best_val, best_k = val, k
    return best_k


def weibull_optimal_prediction(y_hat: float,
                               shape: float = 4.0,
                               n_pts: int = 500) -> float:
    """
    Mitsubishi optimization: trova il valore che minimizza l'E[TWE]
    assumendo che il vero valore segua una Weibull(shape) centrata su y_hat.
    """
    if y_hat <= 0:
        return 0.0

    scale = y_hat / weibull_min.mean(shape)
    t_grid = np.linspace(0, y_hat * 3, n_pts)
    pdf    = weibull_min.pdf(t_grid, shape, scale=scale)

    def expected_twe(y_submit):
        diff = y_submit - t_grid
        w    = np.where(diff >= 0,
                        2.0 / (1.0 + t_grid + 1e-6),
                        1.0 / (1.0 + t_grid + 1e-6))
        return np.trapz(w * diff**2 * pdf, t_grid)

    res = minimize_scalar(expected_twe,
                          bounds=(0, y_hat * 2.5),
                          method='bounded')
    return float(res.x) if res.success else y_hat


def apply_weibull_optimization(y_pred: np.ndarray,
                               shape: float = 4.0) -> np.ndarray:
    """Applica Weibull optimization a ogni predizione."""
    return np.array([weibull_optimal_prediction(v, shape) for v in y_pred])


# ============================================================================
# SEZIONE 1: SENSOR RESIDUALS (Han & Liang)
# ============================================================================

SO_COLS = ['Sensed_Mach', 'Sensed_Altitude', 'Sensed_Pamb',
           'Sensed_TAT',  'Sensed_VAFN',     'Sensed_VBV',
           'Sensed_Fan_Speed', 'Sensed_Pt2']

SD_COLS = ['Sensed_T3', 'Sensed_T45', 'Sensed_Ps3',
           'Sensed_WFuel', 'Sensed_Core_Speed', 'Sensed_T25']


def compute_sensor_residuals(df):
    print("\n" + "="*70)
    print("STEP 1: SENSOR RESIDUALS (per-engine, Ridge regression)")
    print("="*70)

    df = df.copy()
    so_avail = [c for c in SO_COLS if c in df.columns]
    sd_avail = [c for c in SD_COLS if c in df.columns]

    missing = [c for c in SO_COLS+SD_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠️  Missing cols: {missing}")

    df = _fill_nan_per_engine(df, so_avail + sd_avail)

    for esn in sorted(df['ESN'].unique()):
        mask = df['ESN'] == esn
        X_so = df.loc[mask, so_avail].values
        X_so = np.nan_to_num(X_so, nan=0.0)

        valid_mask = None
        for sd_col in sd_avail:
            y_sd    = df.loc[mask, sd_col].values
            res_col = f'{sd_col}_res'

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

    res_cols = [f'{c}_res' for c in sd_avail]
    nan_check = df[res_cols].isna().sum().sum()
    if nan_check:
        print(f"  ⚠️  {nan_check} NaN in residuals → 0")
        df[res_cols] = df[res_cols].fillna(0.0)

    print(f"  Residual cols: {res_cols}")
    return df, res_cols


# ============================================================================
# SEZIONE 2: BASE FEATURES
# ============================================================================

def create_base_features(df):
    print("\n  Creating base features...")
    df = df.copy()

    src = ['Sensed_T45','Sensed_T25','Sensed_Core_Speed','Sensed_WFuel',
           'Sensed_Fan_Speed','Sensed_T3','Sensed_Ps3','Sensed_TAT',
           'Sensed_Pt2','Sensed_Mach']
    df = _fill_nan_per_engine(df, [c for c in src if c in df.columns])

    gamma = 1.4

    df['temp_diff_norm']  = ((df['Sensed_T45'] - df['Sensed_T25']) /
                              (df['Sensed_T25'] + 1e-6)) ** 2
    df['thermal_stress']  = ((df['Sensed_T45'] - df['Sensed_T25']) *
                               df['Sensed_Core_Speed'] / 1000)
    df['load_factor']     = df['Sensed_WFuel'] * df['Sensed_Core_Speed']
    df['speed_ratio']     = df['Sensed_Core_Speed'] / (df['Sensed_Fan_Speed'] + 1e-6)
    df['temp_gradient']   = (df['Sensed_T45'] - df['Sensed_T3']) / (df['Sensed_T3'] + 1e-6)
    df['pressure_ratio']  = df['Sensed_Ps3'] / (df['Sensed_TAT'] + 1e-6)
    df['T45_T3_ratio']    = df['Sensed_T45'] / (df['Sensed_T3'] + 1e-6)
    df['fuel_per_speed']  = df['Sensed_WFuel'] / (df['Sensed_Core_Speed'] + 1e-6)
    df['ps3_pt2_ratio']   = df['Sensed_Ps3'] / (df['Sensed_Pt2'] + 1e-6)

    pr_hpc = (df['Sensed_Ps3'] / (df['Sensed_P25'] + 1e-6)
              if 'Sensed_P25' in df.columns
              else df['Sensed_Ps3'] / (df['Sensed_Pt2'] + 1e-6))
    tr_hpc = df['Sensed_T3'] / (df['Sensed_T25'] + 1e-6)
    df['hptc_efficiency'] = (pr_hpc ** ((gamma-1)/gamma)) / (tr_hpc + 1e-6)
    df['hpt_stress']      = (df['Sensed_Mach'] - df['Sensed_T3']) / (df['Sensed_T45'] + 1e-6)

    # Corrected speeds (Mitsubishi approach)
    df['corrected_fan_speed']  = df['Sensed_Fan_Speed']  / np.sqrt(df['Sensed_TAT'] + 1e-6)
    df['corrected_core_speed'] = df['Sensed_Core_Speed'] / np.sqrt(df['Sensed_TAT'] + 1e-6)

    df = _sanitize(df, "create_base_features")
    print(f"    Created 13 base features")
    return df


# ============================================================================
# SEZIONE 3: AGGREGAZIONE
# ============================================================================

def aggregate_by_cycle(df, res_cols):
    print("\n  Aggregating snapshots → cycles...")

    exclude = ['ESN','Cycles','Snapshot',
               'Cycles_to_WW','Cycles_to_HPC_SV','Cycles_to_HPT_SV']
    feat_cols = [c for c in df.columns if c not in exclude]

    agg = {c: ('median' if c in res_cols else 'mean') for c in feat_cols}
    agg['Cycles_to_WW'] = 'first'

    df_agg = (df.groupby(['ESN','Cycles'])
                .agg(agg)
                .reset_index()
                .sort_values(['ESN','Cycles'])
                .reset_index(drop=True))
    df_agg = _sanitize(df_agg, "aggregate_by_cycle")
    print(f"    {len(df):,} snapshots → {len(df_agg):,} cycles")
    return df_agg


# ============================================================================
# SEZIONE 4: ADAPTIVE WW PERIOD DETECTION (ORIGINALE)
# ============================================================================

def estimate_ww_period_per_engine(df):
    """
    [ORIGINALE] Stima il periodo WW dominante per ogni engine via FFT
    su T45_res detrendato. Nessun paper PHM 2025 fa questo.
    """
    print("\n" + "="*70)
    print("STEP 4a: ADAPTIVE WW PERIOD DETECTION (FFT) [ORIGINALE]")
    print("="*70)

    df = df.copy()
    for esn in sorted(df['ESN'].unique()):
        mask   = df['ESN'] == esn
        df_esn = df[mask].sort_values('Cycles')

        if 'Sensed_T45_res' not in df_esn.columns:
            df.loc[mask, 'ww_period_est']    = 1000.0
            df.loc[mask, 'ww_adaptive_sin']  = 0.0
            df.loc[mask, 'ww_adaptive_cos']  = 0.0
            continue

        sig = df_esn['Sensed_T45_res'].ffill().bfill().fillna(0.0).values
        # Detrend lineare prima della FFT
        x       = np.arange(len(sig))
        coeffs  = np.polyfit(x, sig, 1)
        sig_dt  = sig - np.polyval(coeffs, x)

        freqs, power = periodogram(sig_dt)
        # Cerca il picco nel range [500, 1500] cicli
        valid = (freqs > 1/1500) & (freqs < 1/500) & (freqs > 0)
        if valid.any():
            dom_freq   = freqs[valid][np.argmax(power[valid])]
            period_est = 1.0 / dom_freq
        else:
            period_est = 1000.0

        # Clamp per evitare periodi fisicamente impossibili
        period_est = float(np.clip(period_est, 600, 1400))
        print(f"  ESN {esn}: FFT-estimated WW period = {period_est:.0f} cycles")

        phase = (df_esn['Cycles'].values % period_est) / period_est
        df.loc[mask, 'ww_period_est']   = period_est
        df.loc[mask, 'ww_adaptive_sin'] = np.sin(2 * np.pi * phase)
        df.loc[mask, 'ww_adaptive_cos'] = np.cos(2 * np.pi * phase)

    return df


# ============================================================================
# SEZIONE 5: HPC-WW RECOVERY FEATURE (MathWorks)
# ============================================================================

def add_hpc_ww_recovery_feature(df):
    """
    MathWorks key insight: l'HI HPC mostra un gradino di recupero
    dopo ogni WW event. Feature binaria + cicli dall'ultimo recupero.
    """
    print("\n" + "="*70)
    print("STEP 4b: HPC-WW RECOVERY FEATURE (MathWorks)")
    print("="*70)

    df = df.copy()
    for esn in sorted(df['ESN'].unique()):
        mask   = df['ESN'] == esn
        df_esn = df[mask].sort_values('Cycles').reset_index(drop=True)

        if 'Sensed_T3_res' not in df_esn.columns:
            for col in ['hpc_hi_recovery','cycles_since_ww_recovery',
                        'hpc_hi_slope10','inter_recovery_interval']:
                df.loc[mask, col] = 0.0
            continue

        # Proxy HI HPC = −T3_res (sale con degrado HPC)
        hpc_hi = (-df_esn['Sensed_T3_res'].ffill().bfill().fillna(0.0))
        delta  = hpc_hi.diff().fillna(0.0)

        # Recovery = salto positivo oltre 2.5σ
        thr      = delta.std() * 2.5
        recovery = (delta > thr).astype(float)

        n = len(recovery)
        cycles_since  = np.zeros(n)
        intervals     = np.zeros(n)
        hi_slope      = np.zeros(n)
        rec_times     = []
        last_rec      = 0

        for i in range(n):
            if recovery.iloc[i] == 1:
                rec_times.append(i)
                last_rec = i
            cycles_since[i] = i - last_rec
            if len(rec_times) >= 2:
                intervals[i] = np.mean(np.diff(rec_times[-5:]))
            else:
                intervals[i] = 1000.0

        # Slope locale HPC HI (finestra 10)
        for i in range(n):
            start = max(0, i-9)
            seg   = hpc_hi.iloc[start:i+1].values
            if len(seg) > 2:
                hi_slope[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0]

        df.loc[mask, 'hpc_hi_recovery']          = recovery.values
        df.loc[mask, 'cycles_since_ww_recovery'] = cycles_since
        df.loc[mask, 'hpc_hi_slope10']           = hi_slope
        df.loc[mask, 'inter_recovery_interval']  = intervals

        print(f"  ESN {esn}: {int(recovery.sum())} WW recovery events detected")

    return df


# ============================================================================
# SEZIONE 6: RESIDUAL SHOCK DETECTOR (ORIGINALE)
# ============================================================================

def add_residual_shock_features(df):
    """
    [ORIGINALE] Detecta shock (jump improvvisi) in T45_res.
    Costruisce features predittive per il WW futuro basate
    sull'intervallo inter-shock storico.
    """
    print("\n" + "="*70)
    print("STEP 4c: RESIDUAL SHOCK DETECTOR [ORIGINALE]")
    print("="*70)

    df = df.copy()
    for esn in sorted(df['ESN'].unique()):
        mask   = df['ESN'] == esn
        df_esn = df[mask].sort_values('Cycles').reset_index(drop=True)

        if 'Sensed_T45_res' not in df_esn.columns:
            for col in ['cycles_since_last_shock','shock_magnitude_cumsum',
                        'inter_shock_interval','cycles_to_next_shock_est']:
                df.loc[mask, col] = 0.0
            continue

        sig    = df_esn['Sensed_T45_res'].ffill().bfill().fillna(0.0)
        delta  = sig.diff().fillna(0.0)
        thr    = -delta.std() * 2.5      # drop negativo = WW reset T45_res

        shocks = (delta < thr).values
        n      = len(shocks)

        cs_arr  = np.zeros(n)
        mag_cs  = np.zeros(n)
        int_arr = np.zeros(n)
        nxt_arr = np.zeros(n)

        shock_times = []
        last_shock  = 0
        cum_mag     = 0.0

        for i in range(n):
            if shocks[i]:
                shock_times.append(i)
                cum_mag    += abs(delta.iloc[i])
                last_shock  = i
            cs_arr[i]  = i - last_shock
            mag_cs[i]  = cum_mag
            if len(shock_times) >= 2:
                int_arr[i] = np.mean(np.diff(shock_times[-5:]))
            else:
                int_arr[i] = 1000.0
            nxt_arr[i] = max(int_arr[i] - cs_arr[i], 0.0)

        df.loc[mask, 'cycles_since_last_shock']    = cs_arr
        df.loc[mask, 'shock_magnitude_cumsum']      = mag_cs
        df.loc[mask, 'inter_shock_interval']        = int_arr
        df.loc[mask, 'cycles_to_next_shock_est']    = nxt_arr

        print(f"  ESN {esn}: {int(shocks.sum())} shocks detected, "
              f"avg interval={int_arr[-1]:.0f} cycles")

    return df


# ============================================================================
# SEZIONE 7: PERIODIC + RESIDUAL FEATURES
# ============================================================================

def _compute_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Slope su finestra mobile — fully vectorized con stride_tricks."""
    n = len(arr)
    slopes = np.zeros(n)
    if window < 2 or n < window:
        return slopes
    x = np.arange(window, dtype=float)
    xm = x.mean()
    dx = x - xm
    denom = (dx ** 2).sum()
    if denom < 1e-12:
        return slopes
    # sliding windows senza loop
    shape   = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        arr.astype(float), shape=shape, strides=strides)
    ym = windows.mean(axis=1, keepdims=True)
    slopes[window - 1:] = ((windows - ym) * dx).sum(axis=1) / denom
    return slopes


def add_periodic_and_residual_features(df, res_cols):
    print("\n" + "="*70)
    print("STEP 4d: PERIODIC + RESIDUAL FEATURES")
    print("="*70)

    df = df.copy()
    key_base      = ['temp_gradient','hptc_efficiency','thermal_stress',
                     'T45_T3_ratio','fuel_per_speed',
                     'corrected_core_speed','corrected_fan_speed']
    res_available = [c for c in res_cols if c in df.columns]
    WINDOWS       = [5, 10, 30, 50, 100]

    for esn in sorted(df['ESN'].unique()):
        mask      = df['ESN'] == esn
        df_esn    = df[mask].sort_values('Cycles').reset_index(drop=True)
        max_cycle = df_esn['Cycles'].max()
        n         = len(df_esn)

        # Relative + absolute cycle
        rc = df_esn['Cycles'] / max_cycle
        df.loc[mask, 'relative_cycle']         = rc.values
        df.loc[mask, 'relative_cycle_squared'] = (rc**2).values
        df.loc[mask, 'relative_cycle_cubed']   = (rc**3).values
        df.loc[mask, 'cycles_since_start']     = df_esn['Cycles'].values

        # Phase features (fissi + adattativi già aggiunti da FFT)
        for period in [900, 1000, 1100]:
            phase = (df_esn['Cycles'] % period) / period
            df.loc[mask, f'ww_sin_{period}'] = np.sin(2*np.pi*phase).values
            df.loc[mask, f'ww_cos_{period}'] = np.cos(2*np.pi*phase).values

        # Rolling su residui
        for rcol in res_available:
            sig = df_esn[rcol].copy().ffill().bfill()
            sig = sig.fillna(sig.median() if pd.notna(sig.median()) else 0.0)
            arr = sig.values

            for w in WINDOWS:
                df.loc[mask, f'{rcol}_rmean_{w}'] = sig.rolling(w, min_periods=1).mean().values
                df.loc[mask, f'{rcol}_rstd_{w}']  = sig.rolling(w, min_periods=2).std().fillna(0).values

            df.loc[mask, f'{rcol}_rate']  = sig.diff().fillna(0).values
            df.loc[mask, f'{rcol}_accel'] = sig.diff().diff().fillna(0).values

            cs = sig.cumsum()
            cm = cs.abs().max()
            df.loc[mask, f'{rcol}_cumsum_norm'] = (
                (cs/(cm+1e-6)).values if cm > 1e-6 else np.zeros(n))

            for sw in [20, 50, 100]:
                df.loc[mask, f'{rcol}_slope{sw}'] = _compute_slope(arr, sw)

        # Rolling su base features
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
            cs = sig.cumsum(); cm = cs.abs().max()
            df.loc[mask, f'{feat}_cumsum_norm'] = (
                (cs/(cm+1e-6)).values if cm > 1e-6 else np.zeros(n))

        # Interactions
        if 'Sensed_T45_res' in df_esn.columns:
            t45r = df_esn['Sensed_T45_res'].ffill().bfill().fillna(0.0)
            if 'temp_gradient' in df_esn.columns:
                df.loc[mask, 'T45res_x_tempgrad'] = (
                    t45r * df_esn['temp_gradient'].ffill().bfill().fillna(0.0)).values
            if 'hptc_efficiency' in df_esn.columns:
                df.loc[mask, 'T45res_x_heff'] = (
                    t45r * df_esn['hptc_efficiency'].ffill().bfill().fillna(0.0)).values

        print(f"    ESN {esn}: periodic + residual features added")

    df = _sanitize(df, "add_periodic_and_residual_features")
    print(f"\n  Dataset shape: {df.shape}")
    return df


# ============================================================================
# SEZIONE 8: FEATURE SELECTION
# ============================================================================

def select_features(df, res_cols, top_k=70):
    print("\n" + "="*70)
    print("STEP 5: FEATURE SELECTION (RF importance, top_k={})".format(top_k))
    print("="*70)

    exclude = ['ESN','Cycles','Snapshot',
               'Cycles_to_WW','Cycles_to_HPC_SV','Cycles_to_HPT_SV']
    cands = [c for c in df.columns if c not in exclude]

    X = SimpleImputer(strategy='median').fit_transform(df[cands].values)
    y = df['Cycles_to_WW'].values
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]

    rf = RandomForestRegressor(n_estimators=150, max_depth=15,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)

    imp    = rf.feature_importances_
    ranked = np.argsort(imp)[::-1]
    top    = [cands[i] for i in ranked[:top_k]]

    print(f"  Candidates: {len(cands)} | Selected: {top_k}")
    print(f"  Top 10: {top[:10]}")

    must = [
        # Han & Liang core
        'Sensed_T45_res', 'T45res_slope50', 'T45res_slope20', 'T45res_slope100',
        'T45res_x_tempgrad', 'T45res_x_heff',
        # Cycle info
        'relative_cycle', 'cycles_since_start',
        # Cumsum residui
        'Sensed_T45_res_cumsum_norm', 'Sensed_T3_res_cumsum_norm',
        'Sensed_Ps3_res_cumsum_norm', 'Sensed_Core_Speed_res_cumsum_norm',
        'temp_gradient_cumsum_norm',
        # MathWorks HPC-WW
        'hpc_hi_recovery', 'cycles_since_ww_recovery',
        'hpc_hi_slope10', 'inter_recovery_interval',
        # Originali: shock detector
        'cycles_since_last_shock', 'inter_shock_interval',
        'cycles_to_next_shock_est', 'shock_magnitude_cumsum',
        # Originali: FFT period
        'ww_adaptive_sin', 'ww_adaptive_cos', 'ww_period_est',
    ]
    for col in must:
        if col in df.columns and col not in top:
            top.append(col)
            print(f"  Force-added: {col}")

    return top


# ============================================================================
# SEZIONE 9: LAGGED FEATURES
# ============================================================================

def create_lagged_features(df, feature_cols, window_size=30):
    X, y, info = [], [], []

    for esn in sorted(df['ESN'].unique()):
        df_e = df[df['ESN'] == esn].sort_values('Cycles').reset_index(drop=True)

        for col in feature_cols:
            if col in df_e.columns:
                s = df_e[col].ffill().bfill()
                df_e[col] = s.fillna(s.median() if pd.notna(s.median()) else 0.0)

        for i in range(window_size, len(df_e)):
            win  = df_e.iloc[i-window_size:i]
            feat = []
            for col in feature_cols:
                v = win[col].values.astype(float)
                all_nan = np.isnan(v).all()
                feat.append(np.nanmean(v))
                feat.append(0.0 if all_nan else np.nanstd(v))
                feat.append(0.0 if all_nan else np.nanmin(v))
                feat.append(0.0 if all_nan else np.nanmax(v))
                # Trend lineare nella finestra
                if not all_nan and (~np.isnan(v)).sum() > 2:
                    idx = np.where(~np.isnan(v))[0]
                    feat.append(np.polyfit(idx, v[idx], 1)[0])
                else:
                    feat.append(0.0)
            feat.extend(df_e.iloc[i][feature_cols].fillna(0.0).values)

            X.append(feat)
            y.append(df_e.iloc[i]['Cycles_to_WW'])
            info.append({'ESN': esn, 'Cycle': df_e.iloc[i]['Cycles']})

    X_arr = np.nan_to_num(np.array(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.array(y, float)
    return X_arr, y_arr, pd.DataFrame(info)


# ============================================================================
# SEZIONE 10: XGBoost + LightGBM con custom TWE loss
# ============================================================================

def lgb_twe_obj(y_pred, train_data):
    """Custom TWE objective per LightGBM — parametri ufficiali PHM 2025."""
    y_true = train_data.get_label()
    alpha  = 0.01
    beta   = 1.0 / (y_true.max() + 1e-6)
    diff   = y_pred - y_true
    w      = np.where(diff >= 0,
                      2.0 * alpha / (1.0 + beta * y_true),
                      1.0 * alpha / (1.0 + beta * y_true))
    grad = 2.0 * w * diff
    hess = 2.0 * w
    return grad, hess



def xgb_twe_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    score  = twe(y_true, y_pred)
    return 'twe', score


def xgb_twe_obj(y_pred: np.ndarray, y_true: np.ndarray):
    """Custom TWE objective per XGBoost — parametri ufficiali PHM 2025."""
    alpha = 0.01
    beta  = 1.0 / (y_true.max() + 1e-6)
    diff  = y_pred - y_true
    w     = np.where(diff >= 0,
                     2.0 * alpha / (1.0 + beta * y_true),
                     1.0 * alpha / (1.0 + beta * y_true))
    grad = 2.0 * w * diff
    hess = 2.0 * w
    return grad, hess


def lgb_twe_eval(y_pred, train_data):
    y_true = train_data.get_label()
    score  = twe(y_true, y_pred)
    return 'twe', score, False   # False = lower is better


# ============================================================================
# SEZIONE 11: VARIANCE-AWARE ENSEMBLE + TWE META-LEARNER
# ============================================================================

def _safe_clip(arr):
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0, None)


def train_full_ensemble(X_train, y_train, X_test, y_test,
                        info_train=None, info_test=None):
    print(f"\n  Training Full Ensemble (4 models + TWE meta-learner)...")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)

    y_mean     = y_train.mean()
    y_std      = y_train.std()
    y_test_std = y_test.std()
    n_train    = len(y_train)

    print(f"    Train: mean={y_mean:.0f}, std={y_std:.0f} | "
          f"Test std: {y_test_std:.0f}")

    # ── Inner 3-fold per meta-learner OOF predictions ──────────────────
    kf    = KFold(n_splits=3, shuffle=True, random_state=42)

    #Saltiamo i primi due modelli (RF) e facciamo solo XGB e LGB con TWE loss in modo da ottimizzare
    oof = {m: np.zeros(n_train) for m in ['xgb_twe', 'lgb_twe']}

    preds_test = {}

    # ── Model 1: Normalized RF ──────────────────────────────────────────
    '''print(f"    [1/4] Normalized RF...")
    y_norm = (y_train - y_mean) / (y_std + 1e-6)
    rf1 = RandomForestRegressor(n_estimators=300, max_depth=25,
                                min_samples_leaf=1, random_state=42,
                                n_jobs=-1, verbose=0)
    for tr_idx, val_idx in kf.split(X_train):
        rf1.fit(X_train[tr_idx], y_norm[tr_idx])
        oof['rf_norm'][val_idx] = rf1.predict(X_train[val_idx]) * y_std + y_mean
    rf1.fit(X_train, y_norm)
    p1     = rf1.predict(X_test) * y_std + y_mean
    cap    = 3.0
    ratio  = min(y_test_std / (p1.std() + 1e-6), cap)
    p1_vc  = p1.mean() + (p1 - p1.mean()) * ratio
    preds_test = {'rf_norm': _safe_clip(p1_vc)}
    print(f"      Range: [{preds_test['rf_norm'].min():.0f}, "
          f"{preds_test['rf_norm'].max():.0f}]")

    # ── Model 2: Weighted RF ────────────────────────────────────────────
    print(f"    [2/4] Weighted RF...")
    w2 = np.ones(n_train)
    w2[y_train < np.percentile(y_train, 20)] = 2.0
    w2[y_train > np.percentile(y_train, 80)] = 2.5
    rf2 = RandomForestRegressor(n_estimators=300, max_depth=25,
                                min_samples_leaf=1, random_state=43,
                                n_jobs=-1, verbose=0)
    for tr_idx, val_idx in kf.split(X_train):
        rf2.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=w2[tr_idx])
        oof['rf_weighted'][val_idx] = rf2.predict(X_train[val_idx])
    rf2.fit(X_train, y_train, sample_weight=w2)
    preds_test['rf_weighted'] = _safe_clip(rf2.predict(X_test))
    print(f"      Range: [{preds_test['rf_weighted'].min():.0f}, "
          f"{preds_test['rf_weighted'].max():.0f}]")'''

    # ── Model 3: XGBoost con custom TWE loss ───────────────────────────
    print("    [3/4] XGBoost (MSE debug)...")
    xgb_params = dict(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        objective="reg:squarederror",
    )

    for tr_idx, val_idx in kf.split(X_train):
        m = XGBRegressor(**xgb_params)
        m.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
        oof['xgb_twe'][val_idx] = m.predict(X_train[val_idx])

    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)
    preds_test['xgb_twe'] = _safe_clip(xgb_model.predict(X_test))
    print(f"      Range: [{preds_test['xgb_twe'].min():.0f}, {preds_test['xgb_twe'].max():.0f}]")

    # ── Model 4: LightGBM con custom TWE loss ─────────────────────────
    print(f"    [4/4] LightGBM (TWE loss)...")
    def lgb_twe_eval(y_pred, eval_data):
        y_true = eval_data.get_label()
        score = twe(y_true, y_pred)
        return "twe", score, False

    lgb_params = dict(
        objective=lgb_twe_obj, 
        metric=None,         
        n_estimators=500, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
  
    )
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    train_ds  = lgb.Dataset(X_train, label=y_train)
    valid_ds = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
    for tr_idx, val_idx in kf.split(X_train):
        tr_ds_fold = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
        m_fold = lgb.train(
            {**lgb_params, 'verbose': -1},
            tr_ds_fold,
            feval=lgb_twe_eval,
            num_boost_round=500,
            valid_sets=[lgb.Dataset(X_train[val_idx], label=y_train[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(-1)]
    )
        oof['lgb_twe'][val_idx] = m_fold.predict(X_train[val_idx])

    lgb_final = lgb.train(
    {
        **lgb_params, 'verbose': -1},
        train_ds,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(-1)]
    )
    preds_test['lgb_twe'] = _safe_clip(lgb_final.predict(X_test))
    print(f"      Range: [{preds_test['lgb_twe'].min():.0f}, "
          f"{preds_test['lgb_twe'].max():.0f}]")

    # ── TWE-Weighted Stacking Meta-Learner [ORIGINALE] ─────────────────

    print(f"\n    [META] TWE-Weighted Stacking...")
    '''X_meta_train = np.column_stack([oof['xgb_twe'], oof['lgb_twe']])
    X_meta_test  = np.column_stack([preds_test['xgb_twe'], preds_test['lgb_twe']])

    # Pesi TWE: campioni con y basso (near-event) pesano di più
    meta_w = 2.0 / (1.0 + y_train + 1e-6)
    meta_w = meta_w / meta_w.mean()

    meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
    meta.fit(X_meta_train, y_train, sample_weight=meta_w)

    coef_str = ", ".join(f"{m}={v:.3f}" for m, v in
                         zip(preds_test.keys(), meta.coef_))
    print(f"    Meta-weights: {coef_str}")

    y_meta = _safe_clip(meta.predict(X_meta_test))'''

    # Calcola TWE OOF per ogni modello → chi va meglio pesa di più
    twe_xgb = twe(y_train, oof['xgb_twe'])
    twe_lgb = twe(y_train, oof['lgb_twe'])

    w_xgb = 1.0 / (twe_xgb + 1e-9)
    w_lgb = 1.0 / (twe_lgb + 1e-9)
    tot   = w_xgb + w_lgb
    w_xgb /= tot
    w_lgb /= tot

    print(f"    OOF TWE  → xgb={twe_xgb:.4f}, lgb={twe_lgb:.4f}")
    print(f"    Meta-weights: xgb_twe={w_xgb:.3f}, lgb_twe={w_lgb:.3f}")

    y_meta = _safe_clip(
        w_xgb * preds_test['xgb_twe'] +
        w_lgb * preds_test['lgb_twe']
    )

    # ── Isotonic blend (con check compatibilità cicli) ────────────────
    y_final = y_meta
    '''if info_train is not None and info_test is not None:
        train_max = info_train['Cycle'].max()
        test_max  = info_test['Cycle'].max()
        overlap   = min(train_max, test_max) / (max(train_max, test_max) + 1e-6)
        if overlap > 0.85:
            try:
                iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
                iso.fit(info_train['Cycle'].values, y_train)
                y_iso   = iso.predict(info_test['Cycle'].values)
                y_final = 0.80 * y_meta + 0.20 * y_iso
                print(f"    Isotonic blend (overlap={overlap:.2f}): "
                      f"[{y_iso.min():.0f}, {y_iso.max():.0f}]")
            except Exception as e:
                print(f"    ⚠️  Isotonic failed: {e}")
        else:
            print(f"    Isotonic skipped (cycle overlap={overlap:.2f} < 0.85)")'''

    # ── TWE-Aware Variance Correction [ORIGINALE] ─────────────────────
    # Prima stretch (espande il range)
    opt_k   = twe_optimal_stretch(y_final)
    mu      = y_final.mean()
    y_final = _safe_clip(mu + (y_final - mu) * opt_k)

    # Poi shift (corregge il bias residuo)
    opt_shift = twe_optimal_shift(y_final)
    y_final   = _safe_clip(y_final + opt_shift)
    print(f"    TWE stretch: k={opt_k:.3f}, shift={opt_shift:+.1f}")


    # ── Weibull Optimization (Mitsubishi) ─────────────────────────────
    print(f"    Weibull optimization...")
    y_final = apply_weibull_optimization(y_final, shape=4.0)
    y_final = _safe_clip(np.array(y_final))
    print(f"    Final range: [{y_final.min():.0f}, {y_final.max():.0f}]")

    return y_final


# ============================================================================
# SEZIONE 12: LOEO
# ============================================================================

def run_loeo(df, res_cols, window_size=32, top_k=70):
    esns         = sorted(df['ESN'].unique())
    fold_results = []

    print("\n" + "="*70)
    print(f"STEP 6: LOEO | window={window_size}")
    print("="*70)

    for left_out in esns:
        print(f"\n{'─'*70}")
        print(f"FOLD: Leave out ESN {left_out}")
        print(f"{'─'*70}")

        df_train = df[df['ESN'] != left_out].copy()
        df_test  = df[df['ESN'] == left_out].copy()

        # ── ESN 104: partial train — 60% (era 50% — bug fix) ─────────────
        if left_out == 104:
            cutoff   = df_test['Cycles'].quantile(0.60) 
            df_104_p = df_test[df_test['Cycles'] <= cutoff].copy()
            df_train = pd.concat([df_train, df_104_p], ignore_index=True)
            df_test  = df_test[df_test['Cycles'] > cutoff].copy()
            print(f"  [ESN104] Partial train: first 60% cycles added to train")

        # ── Feature selection SOLO su df_train → NO leakage ──────────────
        feature_cols = select_features(df_train, res_cols, top_k=top_k)
        print(f"  Using {len(feature_cols)} features for LOEO")

        X_train, y_train, info_train = create_lagged_features(
            df_train, feature_cols, window_size)
        X_test,  y_test,  info_test  = create_lagged_features(
            df_test,  feature_cols, window_size)

        if len(X_test) == 0:
            print(f"  ⚠️  No test samples for ESN {left_out}, skip")
            continue

        print(f"  Train: {X_train.shape}, y=[{y_train.min():.0f},{y_train.max():.0f}]")
        print(f"  Test:  {X_test.shape},  y=[{y_test.min():.0f},{y_test.max():.0f}]")

        preprocess = Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scaler',  StandardScaler())])
        X_tr_s = preprocess.fit_transform(X_train)
        X_te_s = preprocess.transform(X_test)

        y_pred = train_full_ensemble(
            X_tr_s, y_train, X_te_s, y_test,
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
            'left_out_esn': left_out,
            'mae':          mae,
            'twe':          twe_sc,
            'r2':           r2,
            'baseline_mae': base,
            'improvement':  improv,
            'y_true':       y_test,
            'y_pred':       y_pred,
            'info_test':    info_test,
        })

    return fold_results


# ============================================================================
# SEZIONE 12b: TRAIN ON ALL + PREDICT TEST
# ============================================================================

def run_full_train_and_predict(df, res_cols, window_size=32, top_k=70):
    """
    Addestra su TUTTI gli ESN e predice su ogni file in data/test_imputed/.
    Output: submission.csv con file,Cycles_to_WW,Cycles_to_HPC_SV,Cycles_to_HPT_SV
    """
    print("\n" + "="*70)
    print("TRAIN ON ALL DATA + PREDICT TEST")
    print("="*70)

    # ── Feature selection su tutti i dati ──────────────────────────────
    feature_cols = select_features(df, res_cols, top_k=top_k)
    print(f"\n  Features selected: {len(feature_cols)}")

    # ── Lagged features su tutto il training ───────────────────────────
    X_train, y_train, info_train = create_lagged_features(df, feature_cols, window_size)
    ok = ~np.isnan(y_train)
    X_train, y_train = X_train[ok], y_train[ok]
    print(f"  Train samples: {len(y_train)}, y=[{y_train.min():.0f}, {y_train.max():.0f}]")

    # ── Preprocessing ──────────────────────────────────────────────────
    preprocess = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    X_tr_s = preprocess.fit_transform(X_train)

    # ── Leggi i file di test ───────────────────────────────────────────
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cartella dello script
    test_dir = os.path.join(BASE_DIR, '..', 'data', 'test_imputed')
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.csv')])
    print(f"\n  Test files found: {len(test_files)}")

    rows = []
    for fname in test_files:
        file_id    = fname.replace('.csv', '')
        df_test_raw = pd.read_csv(os.path.join(test_dir, fname))
        print(f"\n  ── {fname} ({df_test_raw.shape}) ──")

        # Aggiungi ESN fittizio e target dummy se mancanti
        if 'ESN' not in df_test_raw.columns:
            df_test_raw['ESN'] = 9999
        if 'Cycles_to_WW' not in df_test_raw.columns:
            df_test_raw['Cycles_to_WW'] = 0.0

        # Feature engineering identica al train
        df_t, _     = compute_sensor_residuals(df_test_raw)
        df_t        = create_base_features(df_t)
        df_t_agg    = aggregate_by_cycle(df_t, res_cols)
        df_t_agg    = estimate_ww_period_per_engine(df_t_agg)
        df_t_agg    = add_hpc_ww_recovery_feature(df_t_agg)
        df_t_agg    = add_residual_shock_features(df_t_agg)
        df_t_agg    = add_periodic_and_residual_features(df_t_agg, res_cols)

        # Assicura che tutte le colonne esistano
        for col in feature_cols:
            if col not in df_t_agg.columns:
                df_t_agg[col] = 0.0

        X_test, _, _ = create_lagged_features(df_t_agg, feature_cols, window_size)

        if len(X_test) == 0:
            ww_pred = int(round(float(y_train.mean())))
            print(f"  ⚠️  Not enough cycles → fallback mean={ww_pred}")
        else:
            X_te_s  = preprocess.transform(X_test)
            y_p     = train_full_ensemble(X_tr_s, y_train, X_te_s, y_train)
            ww_pred = max(0, int(round(float(y_p[-1]))))   # ultima predizione = ciclo corrente
            print(f"  Cycles_to_WW = {ww_pred}")

        rows.append({
            'file':              file_id,
            'Cycles_to_WW':     ww_pred,
            'Cycles_to_HPC_SV': 0,   # placeholder — da riempire dopo
            'Cycles_to_HPT_SV': 0,   # placeholder — da riempire dopo
        })

    df_sub = pd.DataFrame(rows)
    df_sub.to_csv('submission.csv', index=False)
    print("\n" + "="*70)
    print("SAVED: submission.csv")
    print("="*70)
    print(df_sub.to_string(index=False))
    return df_sub



# ============================================================================
# SEZIONE 13: VISUALIZZAZIONE
# ============================================================================

def plot_results(fold_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for fr in fold_results:
        esn    = fr['left_out_esn']
        cycles = fr['info_test']['Cycle'].values
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(cycles, fr['y_true'], '-o', ms=3, lw=2, alpha=0.85,
                label='Actual', color='#2E86AB')
        ax.plot(cycles, fr['y_pred'], '--', lw=2.5, alpha=0.80,
                label='Predicted', color='#F18F01')
        ax.set_xlabel('Cycle', fontweight='bold')
        ax.set_ylabel('RUL to WW (cycles)', fontweight='bold')
        ax.set_title(f"Hybrid v3 – ESN {esn} | "
                     f"MAE={fr['mae']:.1f}, TWE={fr['twe']:.4f}, R²={fr['r2']:.3f}",
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'loeo_esn{esn}.png'), dpi=150)
        plt.close()

    esns  = [fr['left_out_esn'] for fr in fold_results]
    maes  = [fr['mae']          for fr in fold_results]
    twes  = [fr['twe']          for fr in fold_results]
    r2s   = [fr['r2']           for fr in fold_results]
    imps  = [fr['improvement']  for fr in fold_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, vals, title, ylabel, fmt in [
        (axes[0,0], maes, 'MAE per Engine',    'MAE (cycles)', '{:.1f}'),
        (axes[0,1], twes, 'TWE Score (↓)',     'TWE',          '{:.4f}'),
    ]:
        bars = ax.bar(range(len(esns)), vals, edgecolor='black')
        ax.set_xticks(range(len(esns)))
        ax.set_xticklabels([f'ESN {e}' for e in esns], fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    fmt.format(v), ha='center', va='bottom', fontweight='bold')

    ax = axes[1, 0]
    colors = ['#e74c3c','#3498db','#2ecc71','#f39c12']
    for i, fr in enumerate(fold_results):
        ax.scatter(fr['y_true'], fr['y_pred'], s=15, alpha=0.45,
                   color=colors[i%4], label=f"ESN {fr['left_out_esn']}")
    all_y = np.concatenate([fr['y_true'] for fr in fold_results])
    lim   = [0, all_y.max()*1.05]
    ax.plot(lim, lim, 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual RUL', fontweight='bold')
    ax.set_ylabel('Predicted RUL', fontweight='bold')
    ax.set_title('Actual vs Predicted', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    txt = (f"HYBRID v3 RESULTS\n{'='*38}\n\n"
           f"Avg MAE:  {np.mean(maes):.1f} cycles\n"
           f"Avg TWE:  {np.mean(twes):.4f}\n"
           f"Avg R²:   {np.mean(r2s):.3f}\n"
           f"Avg Impr: {np.mean(imps):+.1f}%\n\n"
           "Per-fold:\n")
    for fr in fold_results:
        txt += (f"  ESN {fr['left_out_esn']}: "
                f"MAE={fr['mae']:.1f}, TWE={fr['twe']:.4f}, "
                f"R²={fr['r2']:.3f}\n")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.7))

    plt.suptitle("WW Prediction – Hybrid v3 (Residual+Periodic+Original)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(os.path.join(out_dir, 'loeo_summary.png'), dpi=150)
    plt.close()
    print(f"\n  ✓ Plots saved → {out_dir}/")


# ============================================================================
# SEZIONE 14: SALVA CSV
# ============================================================================

def save_results(fold_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = [{
        'left_out_esn': fr['left_out_esn'],
        'mae':          fr['mae'],
        'twe':          fr['twe'],
        'r2':           fr['r2'],
        'baseline_mae': fr['baseline_mae'],
        'improvement':  fr['improvement'],
    } for fr in fold_results]

    df_res = pd.DataFrame(rows)
    path   = os.path.join(out_dir, 'hybrid_v3_results.csv')
    df_res.to_csv(path, index=False)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(df_res.to_string(index=False))
    print(f"\n  Avg MAE : {df_res['mae'].mean():.1f}")
    print(f"  Avg TWE : {df_res['twe'].mean():.4f}")
    print(f"  Avg R²  : {df_res['r2'].mean():.3f}")
    print(f"  Avg Impr: {df_res['improvement'].mean():+.1f}%")
    print("="*70)
    return df_res


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("WW PREDICTION – HYBRID v3 (Residual+Periodic+Original)")
    print("="*70)

    cfg = load_config('configs/config.yaml')
    df  = pd.read_csv(cfg['data']['train_clean_csv'])
    print(f"\nLoaded: {df.shape}, ESNs: {sorted(df['ESN'].unique())}")

    # NaN report iniziale
    total_nan = df.isna().sum().sum()
    if total_nan:
        print(f"\n  ⚠️  RAW: {total_nan} NaN found:")
        print(df.isna().sum()[df.isna().sum() > 0].to_string())

    # ── Pipeline ────────────────────────────────────────────────────────
    df, res_cols = compute_sensor_residuals(df)

    print("\n" + "="*70)
    print("STEP 2: BASE FEATURES")
    print("="*70)
    df = create_base_features(df)

    print("\n" + "="*70)
    print("STEP 3: AGGREGATING BY CYCLE")
    print("="*70)
    df_agg = aggregate_by_cycle(df, res_cols)

    # Step 4: feature originali + MathWorks
    df_agg = estimate_ww_period_per_engine(df_agg)   # FFT [ORIGINALE]
    df_agg = add_hpc_ww_recovery_feature(df_agg)     # MathWorks
    df_agg = add_residual_shock_features(df_agg)     # Shock [ORIGINALE]
    df_agg = add_periodic_and_residual_features(df_agg, res_cols)

    # Step 5: feature selection
    # feature_cols = select_features(df_agg, res_cols, top_k=70)
    # print(f"\n  Using {len(feature_cols)} features for LOEO")

    # Step 6: LOEO
    '''window_size  = 30
    fold_results = run_loeo(df_agg, res_cols,      
                            window_size=window_size,
                            top_k=70)

    # Step 7: output
    out_dir = f'artifacts/hybrid_v3_loeo_{window_size}'
    plot_results(fold_results, out_dir)
    save_results(fold_results, out_dir)

    print(f"\n📁 Results: {out_dir}/")
    print("\n" + "="*70)
    print("DONE ✨")
    print("="*70)'''
    run_full_train_and_predict(df_agg, res_cols, window_size=30, top_k=70)


if __name__ == '__main__':
    main()
