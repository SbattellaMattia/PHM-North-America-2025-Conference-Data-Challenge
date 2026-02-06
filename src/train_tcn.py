import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

from src.utils import load_config, set_seed, device
from src.io_paths import ensure_dir
from src.snapshot_scaler import SnapshotStandardScaler
from src.wide_builder import build_wide_per_cycle, wide_feature_columns
from src.dataset_windows import WindowDataset
from src.models_ae import TimeStepAE
from src.models_tcn import TCNRegressor

def loss_fn(name, huber_delta=50.0):
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    return nn.HuberLoss(delta=huber_delta)

@torch.no_grad()
def encode_batch(ae, X):
    ae.eval()
    _, z = ae(X)
    return z

def main(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    dev = device()
    ensure_dir(cfg["data"]["models_dir"])

    id_col = cfg["schema"]["id_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    cyc_train = cfg["schema"]["cycle_train_col"]
    sensors = cfg["schema"]["sensors"]
    targets = cfg["targets"]
    snapshots = cfg["snapshots"]

    # Load scaler + AE
    scaler = SnapshotStandardScaler.load(os.path.join(cfg["data"]["artifacts_dir"], "snapshot_scaler.json"))

    ae_ckpt = torch.load(os.path.join(cfg["data"]["models_dir"], "ae.pt"), map_location="cpu")
    in_dim = ae_ckpt["in_dim"]
    ae = TimeStepAE(in_dim, cfg["ae"]["latent_dim"], cfg["ae"]["hidden_dim"], cfg["ae"]["dropout"])
    ae.load_state_dict(ae_ckpt["state_dict"])
    ae = ae.to(dev)
    for p in ae.parameters():
        p.requires_grad = False

    # Load train
    train = pd.read_csv(cfg["data"]["train_csv"])
    train = train[[id_col, cyc_train, snap_col] + sensors + targets].copy()

    print("NaN in targets:", train[targets].isna().sum().to_dict())
    print("Inf in targets:",
    np.isinf(train[targets].to_numpy()).sum())

    train_std = scaler.transform(train)

    bad = ~np.isfinite(train_std[sensors].to_numpy())
    print("Non-finite in standardized sensors:", bad.sum())

    if bad.any():
        # trova le prime colonne col problema
        bad_cols = np.where(bad.any(axis=0))[0]
        print("Non-finite columns:", [sensors[i] for i in bad_cols[:20]])
        # controlla per snapshot
        for k in snapshots:
            part = train_std[train_std[snap_col] == k][sensors].to_numpy()
            nbad = int((~np.isfinite(part)).sum())
            if nbad > 0:
                print(f"Snapshot {k}: non-finite={nbad}")

    train_wide = build_wide_per_cycle(train_std, id_col, cyc_train, snap_col, sensors, snapshots, cfg["missing_snapshot_fill_value"], target_cols=targets)
    feat_cols = wide_feature_columns(sensors, snapshots)

    ds = WindowDataset(train_wide, feat_cols, targets, id_col, cyc_train,
                       L=cfg["windows"]["L"], stride=cfg["windows"]["stride"], min_cycles=cfg["windows"]["min_cycles"])
    dl = DataLoader(ds, batch_size=cfg["tcn"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    # TCN
    tcn = TCNRegressor(in_dim=cfg["ae"]["latent_dim"],
                       channels=tuple(cfg["tcn"]["channels"]),
                       kernel_size=cfg["tcn"]["kernel_size"],
                       dropout=cfg["tcn"]["dropout"],
                       out_dim=3).to(dev)

    opt = torch.optim.AdamW(tcn.parameters(), lr=cfg["tcn"]["lr"])
    crit = loss_fn(cfg["tcn"]["loss"], cfg["tcn"]["huber_delta"])

    tcn.train()
    for epoch in range(cfg["tcn"]["epochs"]):
        losses = []
        for X, y in tqdm(dl, desc=f"TCN epoch {epoch+1}/{cfg['tcn']['epochs']}"):
            X = X.to(dev)
            y = y.to(dev)

            with torch.no_grad():
                z = encode_batch(ae, X)  # [B,L,latent]

            opt.zero_grad()
            yhat = tcn(z)
            loss = crit(yhat, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"TCN epoch {epoch+1}: loss={sum(losses)/len(losses):.6f}")

    out_path = os.path.join(cfg["data"]["models_dir"], "tcn.pt")
    torch.save({"state_dict": tcn.state_dict(), "cfg": cfg["tcn"]}, out_path)
    print("Saved TCN:", out_path)

if __name__ == "__main__":
    main()
