import numpy as np
import pandas as pd
import json
from typing import List

class SnapshotStandardScaler:
    def __init__(self, sensors: List[str], snapshot_col: str, snapshots: List[int], eps: float = 1e-6,
                 sigma_floor: float = 1e-3):
        self.sensors = sensors
        self.snapshot_col = snapshot_col
        self.snapshots = list(snapshots)
        self.eps = eps
        self.sigma_floor = sigma_floor
        self.mu_ = None
        self.sigma_ = None

    def fit(self, df: pd.DataFrame):
        mu = {s: {} for s in self.sensors}
        sg = {s: {} for s in self.sensors}

        for k in self.snapshots:
            d = df[df[self.snapshot_col] == k]
            for s in self.sensors:
                x = d[s].astype(float).to_numpy()

                m = float(np.nanmean(x))
                v = float(np.nanstd(x))

                if not np.isfinite(m):
                    m = 0.0
                if (not np.isfinite(v)) or v < self.sigma_floor:
                    v = 1.0  # evita esplosioni

                mu[s][k] = m
                sg[s][k] = v

        self.mu_ = mu
        self.sigma_ = sg
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # ripulisci subito eventuali inf nei grezzi
        out[self.sensors] = out[self.sensors].replace([np.inf, -np.inf], np.nan)

        for k in self.snapshots:
            idx = out[self.snapshot_col] == k
            if not idx.any():
                continue
            for s in self.sensors:
                mu = self.mu_[s][k]
                sg = self.sigma_[s][k]
                out.loc[idx, s] = (out.loc[idx, s].astype(float) - mu) / (sg + self.eps)

        # dopo scaling: qualsiasi NaN -> 0 (media nello spazio standardizzato)
        out[self.sensors] = out[self.sensors].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def save(self, path: str):
        payload = {"mu": self.mu_, "sigma": self.sigma_, "sensors": self.sensors, "snapshots": self.snapshots,
                   "snapshot_col": self.snapshot_col, "eps": self.eps, "sigma_floor": self.sigma_floor}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls(payload["sensors"], payload["snapshot_col"], payload["snapshots"],
                  payload["eps"], payload.get("sigma_floor", 1e-3))
        obj.mu_ = {s: {int(k): v for k, v in d.items()} for s, d in payload["mu"].items()}
        obj.sigma_ = {s: {int(k): v for k, v in d.items()} for s, d in payload["sigma"].items()}
        return obj
