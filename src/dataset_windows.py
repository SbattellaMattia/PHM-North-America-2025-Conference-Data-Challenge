import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    """
    Builds windows of length L from a wide per-cycle table.
    Each item: X [L, D], y [3] (label at final time step).
    """
    def __init__(self, wide_df, feature_cols, target_cols, id_col, cycle_col,
                 L=128, stride=10, min_cycles=200):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.id_col = id_col
        self.cycle_col = cycle_col
        self.L = L
        self.stride = stride

        self.items = []
        for esn, g in wide_df.groupby(id_col, sort=False):
            g = g.sort_values(cycle_col).reset_index(drop=True)
            if len(g) < max(min_cycles, L):
                continue
            X = g[feature_cols].astype(np.float32).to_numpy()
            Y = g[target_cols].astype(np.float32).to_numpy()

            # windows ending at t
            for end in range(L-1, len(g), max(1, stride)):
                start = end - (L-1)
                self.items.append((X[start:end+1], Y[end]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        X, y = self.items[idx]
        return torch.from_numpy(X), torch.from_numpy(y)
