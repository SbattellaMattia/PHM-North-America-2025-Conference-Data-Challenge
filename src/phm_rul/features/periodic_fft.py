"""
features/periodic_fft.py — STEP 4a: Adaptive WW period detection via FFT.
[ORIGINALE]
"""
import numpy as np
import pandas as pd
from scipy.signal import periodogram


def estimate_ww_period_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    [ORIGINALE] Stima il periodo WW dominante per ogni engine via FFT
    su T45_res detrendato. Output: ww_period_est, ww_adaptive_sin, ww_adaptive_cos.
    """
    print("\n" + "="*70)
    print("STEP 4a: ADAPTIVE WW PERIOD DETECTION (FFT) [ORIGINALE]")
    print("="*70)

    df = df.copy()
    for esn in sorted(df["ESN"].unique()):
        mask   = df["ESN"] == esn
        df_esn = df[mask].sort_values("Cycles")

        if "Sensed_T45_res" not in df_esn.columns:
            df.loc[mask, "ww_period_est"]    = 1000.0
            df.loc[mask, "ww_adaptive_sin"]  = 0.0
            df.loc[mask, "ww_adaptive_cos"]  = 0.0
            continue

        sig    = df_esn["Sensed_T45_res"].ffill().bfill().fillna(0.0).values
        x      = np.arange(len(sig))
        coeffs = np.polyfit(x, sig, 1)
        sig_dt = sig - np.polyval(coeffs, x)

        freqs, power = periodogram(sig_dt)
        valid = (freqs > 1/1500) & (freqs < 1/500) & (freqs > 0)
        if valid.any():
            dom_freq   = freqs[valid][np.argmax(power[valid])]
            period_est = 1.0 / dom_freq
        else:
            period_est = 1000.0

        period_est = float(np.clip(period_est, 600, 1400))
        print(f"  ESN {esn}: FFT-estimated WW period = {period_est:.0f} cycles")

        phase = (df_esn["Cycles"].values % period_est) / period_est
        df.loc[mask, "ww_period_est"]   = period_est
        df.loc[mask, "ww_adaptive_sin"] = np.sin(2 * np.pi * phase)
        df.loc[mask, "ww_adaptive_cos"] = np.cos(2 * np.pi * phase)

    return df
