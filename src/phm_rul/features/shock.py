"""
features/shock.py — STEP 4c: Residual Shock Detector [ORIGINALE].
"""
import numpy as np
import pandas as pd


def add_residual_shock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [ORIGINALE] Detecta shock improvvisi in T45_res.
    Output: cycles_since_last_shock, shock_magnitude_cumsum,
            inter_shock_interval, cycles_to_next_shock_est.
    """
    print("\n" + "="*70)
    print("STEP 4c: RESIDUAL SHOCK DETECTOR [ORIGINALE]")
    print("="*70)

    df = df.copy()
    for esn in sorted(df["ESN"].unique()):
        mask   = df["ESN"] == esn
        df_esn = df[mask].sort_values("Cycles").reset_index(drop=True)

        if "Sensed_T45_res" not in df_esn.columns:
            for col in ["cycles_since_last_shock","shock_magnitude_cumsum",
                        "inter_shock_interval","cycles_to_next_shock_est"]:
                df.loc[mask, col] = 0.0
            continue

        sig    = df_esn["Sensed_T45_res"].ffill().bfill().fillna(0.0)
        delta  = sig.diff().fillna(0.0)
        thr    = -delta.std() * 2.5
        shocks = (delta < thr).values
        n      = len(shocks)

        cs_arr  = np.zeros(n); mag_cs  = np.zeros(n)
        int_arr = np.zeros(n); nxt_arr = np.zeros(n)
        shock_times = []; last_shock = 0; cum_mag = 0.0

        for i in range(n):
            if shocks[i]:
                shock_times.append(i)
                cum_mag   += abs(delta.iloc[i])
                last_shock = i
            cs_arr[i]  = i - last_shock
            mag_cs[i]  = cum_mag
            int_arr[i] = (np.mean(np.diff(shock_times[-5:]))
                          if len(shock_times) >= 2 else 1000.0)
            nxt_arr[i] = max(int_arr[i] - cs_arr[i], 0.0)

        df.loc[mask, "cycles_since_last_shock"]  = cs_arr
        df.loc[mask, "shock_magnitude_cumsum"]   = mag_cs
        df.loc[mask, "inter_shock_interval"]     = int_arr
        df.loc[mask, "cycles_to_next_shock_est"] = nxt_arr

        print(f"  ESN {esn}: {int(shocks.sum())} shocks detected, "
              f"avg interval={int_arr[-1]:.0f} cycles")

    return df
