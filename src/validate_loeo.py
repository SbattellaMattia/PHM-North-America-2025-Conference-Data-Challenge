import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_config, set_seed, device
from src.io_paths import ensure_dir
from src.splits import loeo_folds
from src.snapshot_scaler import SnapshotStandardScaler
from src.wide_builder import build_wide_per_cycle, wide_feature_columns
from src.dataset_windows import WindowDataset
from src.models_ae import TimeStepAE
from src.models_tcn import TCNRegressor
from src.eval_metrics import mae, rmse

def predict_dataset(ae, tcn, dl, dev):
    ae.eval(); tcn.eval()
    ys, yh = [], []
    with torch.no_grad():
        for X, y in dl:
            X = X.to(dev)
            y = y.to(dev)
            _, z = ae(X)
            pred = tcn(z)
            ys.append(y.cpu().numpy())
            yh.append(pred.cpu().numpy())
    return np.vstack(ys), np.vstack(yh)

def main(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    dev = device()

    ensure_dir(cfg["data"]["artifacts_dir"])

    # Load full train
    df = pd.read_csv(cfg["data"]["train_csv"])
    id_col = cfg["schema"]["id_col"]
    cyc_col = cfg["schema"]["cycle_train_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    sensors = cfg["schema"]["sensors"]
    targets = cfg["targets"]
    snapshots = cfg["snapshots"]

    df = df[[id_col, cyc_col, snap_col] + sensors + targets].copy()

    folds = list(loeo_folds(df, id_col))
    print("ESNs:", sorted(df[id_col].unique().tolist()))
    print("Num folds:", len(folds))

    all_scores = []

    for fold_idx, (train_esns, val_esns) in enumerate(folds, 1):
        print(f"\n=== Fold {fold_idx}: train={train_esns} val={val_esns} ===")
        dtr = df[df[id_col].isin(train_esns)].copy()
        dva = df[df[id_col].isin(val_esns)].copy()

        # Fit scaler on TRAIN only
        scaler = SnapshotStandardScaler(sensors, snap_col, snapshots).fit(dtr)
        dtr_std = scaler.transform(dtr)
        dva_std = scaler.transform(dva)

        # Wide tables (train includes targets)
        tr_wide = build_wide_per_cycle(
            dtr_std, id_col, cyc_col, snap_col, sensors, snapshots,
            cfg["missing_snapshot_fill_value"], target_cols=targets
        )
        va_wide = build_wide_per_cycle(
            dva_std, id_col, cyc_col, snap_col, sensors, snapshots,
            cfg["missing_snapshot_fill_value"], target_cols=targets
        )

        feat_cols = wide_feature_columns(sensors, snapshots)

        ds_tr = WindowDataset(tr_wide, feat_cols, targets, id_col, cyc_col,
                              L=cfg["windows"]["L"], stride=cfg["windows"]["stride"], min_cycles=cfg["windows"]["min_cycles"])
        ds_va = WindowDataset(va_wide, feat_cols, targets, id_col, cyc_col,
                              L=cfg["windows"]["L"], stride=cfg["windows"]["stride"], min_cycles=cfg["windows"]["min_cycles"])

        dl_tr = DataLoader(ds_tr, batch_size=cfg["ae"]["batch_size"], shuffle=True, num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=cfg["tcn"]["batch_size"], shuffle=False, num_workers=0)

        # ---- Train AE (quick) on fold train
        in_dim = len(feat_cols)
        ae = TimeStepAE(in_dim, cfg["ae"]["latent_dim"], cfg["ae"]["hidden_dim"], cfg["ae"]["dropout"]).to(dev)
        opt_ae = torch.optim.AdamW(ae.parameters(), lr=cfg["ae"]["lr"])
        crit_ae = torch.nn.SmoothL1Loss()
        ae.train()
        for ep in range(cfg["ae"]["epochs"]):
            losses = []
            for X, _ in dl_tr:
                X = X.to(dev)
                Xn = X + cfg["ae"]["noise_std"] * torch.randn_like(X)
                opt_ae.zero_grad()
                Xhat, _ = ae(Xn)
                loss = crit_ae(Xhat, X)
                loss.backward()
                opt_ae.step()
                losses.append(loss.item())
            print(f"AE ep {ep+1}/{cfg['ae']['epochs']} loss={sum(losses)/len(losses):.6f}")

        # ---- Train TCN on fold train (freeze AE)
        for p in ae.parameters():
            p.requires_grad = False

        ds_tr2 = DataLoader(ds_tr, batch_size=cfg["tcn"]["batch_size"], shuffle=True, num_workers=0)
        tcn = TCNRegressor(in_dim=cfg["ae"]["latent_dim"],
                           channels=tuple(cfg["tcn"]["channels"]),
                           kernel_size=cfg["tcn"]["kernel_size"],
                           dropout=cfg["tcn"]["dropout"],
                           out_dim=3).to(dev)
        opt_t = torch.optim.AdamW(tcn.parameters(), lr=cfg["tcn"]["lr"])
        crit = torch.nn.HuberLoss(delta=cfg["tcn"]["huber_delta"]) if cfg["tcn"]["loss"] == "huber" else \
               (torch.nn.MSELoss() if cfg["tcn"]["loss"] == "mse" else torch.nn.L1Loss())

        tcn.train()
        for ep in range(cfg["tcn"]["epochs"]):
            losses = []
            for X, y in ds_tr2:
                X = X.to(dev); y = y.to(dev)
                with torch.no_grad():
                    _, z = ae(X)
                opt_t.zero_grad()
                pred = tcn(z)
                loss = crit(pred, y)
                loss.backward()
                opt_t.step()
                losses.append(loss.item())
            if (ep+1) % 20 == 0 or ep == 0:
                print(f"TCN ep {ep+1}/{cfg['tcn']['epochs']} loss={sum(losses)/len(losses):.3f}")

        # ---- Validate on held-out ESN
        y_true, y_pred = predict_dataset(ae, tcn, dl_va, dev)
        fold_res = {}
        for j, t in enumerate(targets):
            fold_res[t] = {"MAE": mae(y_true[:, j], y_pred[:, j]),
                           "RMSE": rmse(y_true[:, j], y_pred[:, j])}
            print(f"{t}: MAE={fold_res[t]['MAE']:.2f} RMSE={fold_res[t]['RMSE']:.2f}")

        all_scores.append(fold_res)

    print("\nDone LOEO.")
    # (se vuoi, salviamo all_scores su JSON)

if __name__ == "__main__":
    main()
