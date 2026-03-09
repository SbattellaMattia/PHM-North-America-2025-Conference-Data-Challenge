"""
features/base.py — STEP 2: Base engineered features.
"""
import numpy as np
import pandas as pd
from ..common import fill_nan_per_engine, sanitize


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Creating base features...")
    df  = df.copy()
    src = ["Sensed_T45","Sensed_T25","Sensed_Core_Speed","Sensed_WFuel",
           "Sensed_Fan_Speed","Sensed_T3","Sensed_Ps3","Sensed_TAT",
           "Sensed_Pt2","Sensed_Mach"]
    df  = fill_nan_per_engine(df, [c for c in src if c in df.columns])

    gamma = 1.4
    df["temp_diff_norm"]  = ((df["Sensed_T45"] - df["Sensed_T25"]) / (df["Sensed_T25"] + 1e-6)) ** 2
    df["thermal_stress"]  = (df["Sensed_T45"] - df["Sensed_T25"]) * df["Sensed_Core_Speed"] / 1000
    df["load_factor"]     = df["Sensed_WFuel"] * df["Sensed_Core_Speed"]
    df["speed_ratio"]     = df["Sensed_Core_Speed"] / (df["Sensed_Fan_Speed"] + 1e-6)
    df["temp_gradient"]   = (df["Sensed_T45"] - df["Sensed_T3"]) / (df["Sensed_T3"] + 1e-6)
    df["pressure_ratio"]  = df["Sensed_Ps3"] / (df["Sensed_TAT"] + 1e-6)
    df["T45_T3_ratio"]    = df["Sensed_T45"] / (df["Sensed_T3"] + 1e-6)
    df["fuel_per_speed"]  = df["Sensed_WFuel"] / (df["Sensed_Core_Speed"] + 1e-6)
    df["ps3_pt2_ratio"]   = df["Sensed_Ps3"] / (df["Sensed_Pt2"] + 1e-6)

    pr_hpc = (df["Sensed_Ps3"] / (df["Sensed_P25"] + 1e-6)
              if "Sensed_P25" in df.columns
              else df["Sensed_Ps3"] / (df["Sensed_Pt2"] + 1e-6))
    tr_hpc = df["Sensed_T3"] / (df["Sensed_T25"] + 1e-6)
    df["hptc_efficiency"]      = (pr_hpc ** ((gamma-1)/gamma)) / (tr_hpc + 1e-6)
    df["hpt_stress"]           = (df["Sensed_Mach"] - df["Sensed_T3"]) / (df["Sensed_T45"] + 1e-6)
    df["corrected_fan_speed"]  = df["Sensed_Fan_Speed"]  / np.sqrt(df["Sensed_TAT"] + 1e-6)
    df["corrected_core_speed"] = df["Sensed_Core_Speed"] / np.sqrt(df["Sensed_TAT"] + 1e-6)

    df = sanitize(df, "create_base_features")
    print("    Created 13 base features")
    return df
