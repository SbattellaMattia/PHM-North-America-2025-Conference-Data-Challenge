"""
features/recovery.py — STEP 4b: HPC-WW Recovery Feature (MathWorks).
"""
import numpy as np
import pandas as pd


def add_hpc_ww_recovery_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    MathWorks key insight: l'HI HPC mostra un gradino di recupero dopo ogni WW event.
    Output: hpc_hi_recovery, cycles_since_ww_recovery, hpc_hi_slope10, inter_recovery_interval.
    """
    print("\n" + "="*70)
    print("STEP 4b: HPC-WW RECOVERY FEATURE (MathWorks)")
    print("="*70)

    df = df.copy()
    for esn in sorted(df["ESN"].unique()):
        mask   = df["ESN"] == esn
        df_esn = df[mask].sort_values("Cycles").reset_index(drop=True)

        if "Sensed_T3_res" not in df_esn.columns:
            for col in ["hpc_hi_recovery","cycles_since_ww_recovery",
                        "hpc_hi_slope10","inter_recovery_interval"]:
                df.loc[mask, col] = 0.0
            continue

        hpc_hi = -df_esn["Sensed_T3_res"].ffill().bfill().fillna(0.0)
        delta  = hpc_hi.diff().fillna(0.0)
        thr    = delta.std() * 2.5
        recovery = (delta > thr).astype(float)

        n = len(recovery)
        cycles_since = np.zeros(n)
        intervals    = np.zeros(n)
        hi_slope     = np.zeros(n)
        rec_times    = []
        last_rec     = 0

        for i in range(n):
            if recovery.iloc[i] == 1:
                rec_times.append(i)
                last_rec = i
            cycles_since[i] = i - last_rec
            intervals[i] = (np.mean(np.diff(rec_times[-5:]))
                            if len(rec_times) >= 2 else 1000.0)

        for i in range(n):
            start = max(0, i-9)
            seg   = hpc_hi.iloc[start:i+1].values
            if len(seg) > 2:
                hi_slope[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0]

        df.loc[mask, "hpc_hi_recovery"]          = recovery.values
        df.loc[mask, "cycles_since_ww_recovery"] = cycles_since
        df.loc[mask, "hpc_hi_slope10"]           = hi_slope
        df.loc[mask, "inter_recovery_interval"]  = intervals

        print(f"  ESN {esn}: {int(recovery.sum())} WW recovery events detected")

    return df
