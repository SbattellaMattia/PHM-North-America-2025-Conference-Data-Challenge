"""
features/aggregate.py — STEP 3: Aggregazione snapshot → cicli.
"""
import pandas as pd
from ..common import sanitize


def aggregate_by_cycle(df: pd.DataFrame, res_cols: list) -> pd.DataFrame:
    print("\n  Aggregating snapshots → cycles...")
    exclude   = ["ESN","Cycles","Snapshot",
                 "Cycles_to_WW","Cycles_to_HPC_SV","Cycles_to_HPT_SV"]
    feat_cols = [c for c in df.columns if c not in exclude]
    agg       = {c: ("median" if c in res_cols else "mean") for c in feat_cols}
    agg["Cycles_to_WW"] = "first"

    df_agg = (df.groupby(["ESN","Cycles"])
                .agg(agg)
                .reset_index()
                .sort_values(["ESN","Cycles"])
                .reset_index(drop=True))
    df_agg = sanitize(df_agg, "aggregate_by_cycle")
    print(f"    {len(df):,} snapshots → {len(df_agg):,} cycles")
    return df_agg
