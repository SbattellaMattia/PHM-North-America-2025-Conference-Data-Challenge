import numpy as np
import pandas as pd
from typing import List

def build_wide_per_cycle(
    df: pd.DataFrame,
    id_col: str,
    cycle_col: str,
    snapshot_col: str,
    sensors: List[str],
    snapshots: List[int],
    fill_value: float = 0.0,
    target_cols: List[str] = None,
):
    """
    Input: long df with rows per (id, cycle, snapshot).
    Output: wide df with one row per (id, cycle):
      columns: for each snapshot k and sensor s -> f"{s}_s{k}"
      plus mask columns: f"mask_s{k}" (1 if snapshot present else 0)
    """
    key = [id_col, cycle_col]
    wide_rows = []
    groups = df.groupby(key, sort=False)

    for (esn, cyc), g in groups:
        row = {id_col: esn, cycle_col: cyc}

        if target_cols:
            first = g.iloc[0]
            for t in target_cols:
                row[t] = float(first[t])

        present = set(g[snapshot_col].astype(int).tolist())

        for k in snapshots:
            row[f"mask_s{k}"] = 1.0 if k in present else 0.0
            gk = g[g[snapshot_col] == k]
            if len(gk) == 1:
                for s in sensors:
                    row[f"{s}_s{k}"] = float(gk.iloc[0][s])
            elif len(gk) > 1:
                # if duplicated snapshot entries, take median
                for s in sensors:
                    row[f"{s}_s{k}"] = float(np.median(gk[s].astype(float).to_numpy()))
            else:
                for s in sensors:
                    row[f"{s}_s{k}"] = float(fill_value)

        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    wide = wide.sort_values([id_col, cycle_col]).reset_index(drop=True)
    return wide

def wide_feature_columns(sensors: List[str], snapshots: List[int]):
    cols = []
    for k in snapshots:
        cols.extend([f"{s}_s{k}" for s in sensors])
    cols.extend([f"mask_s{k}" for k in snapshots])
    return cols
