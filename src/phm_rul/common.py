"""
common.py — Utility condivise: NaN handling, TWE, Weibull, shift.
Nessuna dipendenza interna al package.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats    import weibull_min


# ── NaN helpers ──────────────────────────────────────────────────────────────

def fill_nan_per_engine(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for esn in df["ESN"].unique():
        mask = df["ESN"] == esn
        for col in cols:
            if col not in df.columns:
                continue
            s = df.loc[mask, col].ffill().bfill()
            if s.isna().any():
                gm = df[col].median()
                s  = s.fillna(gm if pd.notna(gm) else 0.0)
            df.loc[mask, col] = s
    return df


def sanitize(df: pd.DataFrame, ctx: str = "") -> pd.DataFrame:
    df  = df.replace([np.inf, -np.inf], np.nan)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].median()).fillna(0.0)
    left = df.isna().sum().sum()
    if left:
        print(f"  ⚠️  [{ctx}] {left} NaN → 0")
        df = df.fillna(0.0)
    return df


def safe_clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0, None)


# ── TWE ──────────────────────────────────────────────────────────────────────

def twe(y_true: np.ndarray, y_pred: np.ndarray,
        alpha: float = 0.01, beta: float = None) -> float:
    """TWE normalizzato — formula ufficiale PHM 2025."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if beta is None:
        beta = 1.0 / (y_true.max() + 1e-6)

    diff = y_pred - y_true
    w    = np.where(diff >= 0,
                    2.0 * alpha / (1.0 + beta * y_true),
                    1.0 * alpha / (1.0 + beta * y_true))
    return float(np.mean(w * diff**2))


def twe_optimal_shift(y_pred: np.ndarray,
                      search_range: float = 400,
                      n_points: int = 300) -> float:
    """[ORIGINALE] Bias additivo che minimizza il TWE atteso."""
    shifts = np.linspace(-search_range, search_range, n_points)
    best_shift, best_val = 0.0, np.inf
    for s in shifts:
        yp  = np.clip(y_pred + s, 0, None)
        val = twe(y_pred, yp)
        if val < best_val:
            best_val, best_shift = val, s
    return best_shift


# ── Weibull ───────────────────────────────────────────────────────────────────

def weibull_optimal_prediction(y_hat: float,
                               shape: float = 4.0,
                               n_pts: int = 500) -> float:
    """Mitsubishi: minimizza E[TWE] assumendo Weibull(shape) centrata su y_hat."""
    if y_hat <= 0:
        return 0.0
    scale  = y_hat / weibull_min.mean(shape)
    t_grid = np.linspace(0, y_hat * 3, n_pts)
    pdf    = weibull_min.pdf(t_grid, shape, scale=scale)

    def expected_twe(y_submit):
        diff = y_submit - t_grid
        w    = np.where(diff >= 0,
                        2.0 / (1.0 + t_grid + 1e-6),
                        1.0 / (1.0 + t_grid + 1e-6))
        return np.trapz(w * diff**2 * pdf, t_grid)

    res = minimize_scalar(expected_twe, bounds=(0, y_hat * 2.5), method="bounded")
    return float(res.x) if res.success else y_hat


def apply_weibull_optimization(y_pred: np.ndarray,
                               shape: float = 4.0) -> np.ndarray:
    return np.array([weibull_optimal_prediction(v, shape) for v in y_pred])
