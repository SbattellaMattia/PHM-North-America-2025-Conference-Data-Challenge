"""
models/objectives.py — Custom TWE loss per XGBoost e LightGBM.
"""
import numpy as np
from ..common import twe


def xgb_twe_obj(y_pred: np.ndarray, y_true: np.ndarray):
    """Custom TWE objective per XGBRegressor (sklearn API)."""
    diff = y_pred - y_true
    w    = np.where(diff >= 0,
                    2.0 / (1.0 + y_true + 1e-6),
                    1.0 / (1.0 + y_true + 1e-6))
    return 2.0 * w * diff, 2.0 * w


def lgb_twe_obj(y_pred: np.ndarray, train_data):
    """Custom TWE objective per LightGBM."""
    y_true = train_data.get_label()
    diff   = y_pred - y_true
    w      = np.where(diff >= 0,
                      2.0 / (1.0 + y_true + 1e-6),
                      1.0 / (1.0 + y_true + 1e-6))
    return 2.0 * w * diff, 2.0 * w * np.ones_like(diff)


def lgb_twe_eval(y_pred: np.ndarray, train_data):
    y_true = train_data.get_label()
    return "twe", twe(y_true, y_pred), False
