import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_config, set_seed, device
from src.io_paths import ensure_dir
from src.snapshot_scaler import SnapshotStandardScaler
from src.wide_builder import build_wide_per_cycle, wide_feature_columns
from src.dataset_windows import WindowDataset
from src.models_ae import TimeStepAE

import pandas as pd

def main(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    dev = device()

    ensure_dir(cfg["data"]["artifacts_dir"])
    ensure_dir(cfg["data"]["models_dir"])

    # Load train
    train = pd.read_csv(cfg["data"]["train_csv"])

    # Keep only columns needed for val/test schema + targets
    id_col = cfg["schema"]["id_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    cyc_train = cfg["schema"]["cycle_train_col"]
    sensors = cfg["schema"]["sensors"]
    targets = cfg["targets"]
    snapshots = cfg["snapshots"]

    train = train[[id_col, cyc_train, snap_col] + sensors + targets].copy()

    # Fit snapshot scaler on train
    scaler = SnapshotStandardScaler(sensors, snap_col, snapshots).fit(train)
    scaler_path = os.path.join(cfg["data"]["artifacts_dir"], "snapshot_scaler.json")
    scaler.save(scaler_path)

    # Standardize
    train_std = scaler.transform(train)

    train_std[sensors] = train_std[sensors].replace([np.inf, -np.inf], np.nan)
    train_std[sensors] = train_std[sensors].fillna(0.0)


    # Build wide per cycle
    train_wide = build_wide_per_cycle(
        train_std, id_col=id_col, cycle_col=cyc_train, snapshot_col=snap_col,
        sensors=sensors, snapshots=snapshots, fill_value=cfg["missing_snapshot_fill_value"], target_cols=targets
    )

    feat_cols = wide_feature_columns(sensors, snapshots)
    # AE is unsupervised: we don't need labels but dataset requires something; pass targets anyway and ignore y
    ds = WindowDataset(
        train_wide, feature_cols=feat_cols, target_cols=targets,
        id_col=id_col, cycle_col=cyc_train,
        L=cfg["windows"]["L"], stride=cfg["windows"]["stride"], min_cycles=cfg["windows"]["min_cycles"]
    )
    dl = DataLoader(ds, batch_size=cfg["ae"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    in_dim = len(feat_cols)
    model = TimeStepAE(in_dim, cfg["ae"]["latent_dim"], cfg["ae"]["hidden_dim"], cfg["ae"]["dropout"]).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ae"]["lr"])
    crit = nn.SmoothL1Loss()  # robust reconstruction loss

    noise_std = cfg["ae"]["noise_std"]
    model.train()
    for epoch in range(cfg["ae"]["epochs"]):
        losses = []
        for X, _ in tqdm(dl, desc=f"AE epoch {epoch+1}/{cfg['ae']['epochs']}"):
            X = X.to(dev)
            Xn = X + noise_std * torch.randn_like(X)

            opt.zero_grad()
            Xhat, _ = model(Xn)
            loss = crit(Xhat, X)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"AE epoch {epoch+1}: loss={sum(losses)/len(losses):.6f}")

    out_path = os.path.join(cfg["data"]["models_dir"], "ae.pt")
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "cfg": cfg["ae"]}, out_path)
    print("Saved AE:", out_path)
    print("Saved scaler:", scaler_path)

if __name__ == "__main__":
    main()
