import numpy as np
import pandas as pd
import json
from typing import Dict, List

class SnapshotStandardScaler:
    """
    Fit mu/sigma per (sensor, snapshot) on TRAIN only.
    Transform any dataframe with columns [Snapshot] + sensors.
    """
    def __init__(self, sensors: List[str], snapshot_col: str, snapshots: List[int], eps: float = 1e-6):
        self.sensors = sensors
        self.snapshot_col = snapshot_col
        self.snapshots = list(snapshots)
        self.eps = eps
        self.mu_ = None
        self.sigma_ = None

    def fit(self, df: pd.DataFrame):
        mu = {s: {} for s in self.sensors}
        sg = {s: {} for s in self.sensors}

        for k in self.snapshots:
            d = df[df[self.snapshot_col] == k]
            for s in self.sensors:
                x = d[s].astype(float).to_numpy()
                mu[s][k] = float(np.nanmean(x))
                sg[s][k] = float(np.nanstd(x))

                sg_val = float(np.nanstd(x))
                if not np.isfinite(sg_val) or sg_val < 1e-3:
                    sg_val = 1.0
                sg[s][k] = sg_val


        self.mu_ = mu
        self.sigma_ = sg
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for k in self.snapshots:
            idx = out[self.snapshot_col] == k
            for s in self.sensors:
                mu = self.mu_[s][k]
                sg = self.sigma_[s][k]
                out.loc[idx, s] = (out.loc[idx, s].astype(float) - mu) / (sg + self.eps)
        return out

    def save(self, path: str):
        payload = {"mu": self.mu_, "sigma": self.sigma_, "sensors": self.sensors, "snapshots": self.snapshots,
                   "snapshot_col": self.snapshot_col, "eps": self.eps}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls(payload["sensors"], payload["snapshot_col"], payload["snapshots"], payload["eps"])
        obj.mu_ = payload["mu"]
        obj.sigma_ = payload["sigma"]
        # json keys become strings -> convert snapshot keys back to int
        obj.mu_ = {s: {int(k): v for k, v in d.items()} for s, d in obj.mu_.items()}
        obj.sigma_ = {s: {int(k): v for k, v in d.items()} for s, d in obj.sigma_.items()}
        return obj
import numpy as np
import pandas as pd
import json
from typing import Dict, List

class SnapshotStandardScaler:
    """
    Fit mu/sigma per (sensor, snapshot) on TRAIN only.
    Transform any dataframe with columns [Snapshot] + sensors.
    """
    def __init__(self, sensors: List[str], snapshot_col: str, snapshots: List[int], eps: float = 1e-6):
        self.sensors = sensors
        self.snapshot_col = snapshot_col
        self.snapshots = list(snapshots)
        self.eps = eps
        self.mu_ = None
        self.sigma_ = None

    def fit(self, df: pd.DataFrame):
        mu = {s: {} for s in self.sensors}
        sg = {s: {} for s in self.sensors}

        for k in self.snapshots:
            d = df[df[self.snapshot_col] == k]
            for s in self.sensors:
                x = d[s].astype(float).to_numpy()
                mu[s][k] = float(np.nanmean(x))
                sg[s][k] = float(np.nanstd(x))

        self.mu_ = mu
        self.sigma_ = sg
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for k in self.snapshots:
            idx = out[self.snapshot_col] == k
            for s in self.sensors:
                mu = self.mu_[s][k]
                sg = self.sigma_[s][k]
                out.loc[idx, s] = (out.loc[idx, s].astype(float) - mu) / (sg + self.eps)
        return out

    def save(self, path: str):
        payload = {"mu": self.mu_, "sigma": self.sigma_, "sensors": self.sensors, "snapshots": self.snapshots,
                   "snapshot_col": self.snapshot_col, "eps": self.eps}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls(payload["sensors"], payload["snapshot_col"], payload["snapshots"], payload["eps"])
        obj.mu_ = payload["mu"]
        obj.sigma_ = payload["sigma"]
        # json keys become strings -> convert snapshot keys back to int
        obj.mu_ = {s: {int(k): v for k, v in d.items()} for s, d in obj.mu_.items()}
        obj.sigma_ = {s: {int(k): v for k, v in d.items()} for s, d in obj.sigma_.items()}
        return obj
