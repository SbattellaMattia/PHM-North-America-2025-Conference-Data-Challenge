import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import load_config, set_seed, device
from src.splits import loeo_folds
from src.snapshot_scaler import SnapshotStandardScaler
from src.wide_builder import build_wide_per_cycle, wide_feature_columns
from src.dataset_windows import WindowDataset
from src.models_ae import TimeStepAE
from src.models_tcn import TCNRegressor
from src.eval_metrics import mae, rmse

@torch.no_grad()
def evaluate_fold(ae, tcn, dl, dev):
    ae.eval(); tcn.eval()
    ys, yh = [], []
    for X, y in dl:
        X, y = X.to(dev), y.to(dev)
        _, z = ae(X)
        pred = tcn(z)
        ys.append(y.cpu().numpy())
        yh.append(pred.cpu().numpy())
    return np.vstack(ys), np.vstack(yh)

def main(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    dev = device()
    
    df = pd.read_csv(cfg["data"]["train_csv"])
    id_col = cfg["schema"]["id_col"]
    cyc_col = cfg["schema"]["cycle_train_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    sensors = cfg["schema"]["sensors"]
    targets = cfg["targets"]
    snapshots = cfg["snapshots"]
    
    df = df[[id_col, cyc_col, snap_col] + sensors + targets].copy()
    folds = list(loeo_folds(df, id_col))
    
    all_scores = []
    
    for fold_idx, (train_esns, val_esns) in enumerate(folds, 1):
        print(f"\n=== Fold {fold_idx}: val={val_esns} ===")
        
        # Load models
        ae_ckpt = torch.load(os.path.join(cfg["data"]["models_dir"], f"ae_fold{fold_idx}.pt"), map_location="cpu", weights_only=False)
        tcn_ckpt = torch.load(os.path.join(cfg["data"]["models_dir"], f"tcn_fold{fold_idx}.pt"), map_location="cpu", weights_only=False)
        
        ae = TimeStepAE(ae_ckpt["in_dim"], cfg["ae"]["latent_dim"], cfg["ae"]["hidden_dim"], cfg["ae"]["dropout"])
        ae.load_state_dict(ae_ckpt["state_dict"])
        ae = ae.to(dev).eval()
        
        tcn = TCNRegressor(in_dim=cfg["ae"]["latent_dim"], channels=tuple(cfg["tcn"]["channels"]),
                           kernel_size=cfg["tcn"]["kernel_size"], dropout=cfg["tcn"]["dropout"], out_dim=3)
        tcn.load_state_dict(tcn_ckpt["state_dict"])
        tcn = tcn.to(dev).eval()
        
        # Prepare data
        dtr = df[df[id_col].isin(train_esns)].copy()
        dva = df[df[id_col].isin(val_esns)].copy()
        
        scaler = SnapshotStandardScaler(sensors, snap_col, snapshots).fit(dtr)
        dva_std = scaler.transform(dva)
        
        va_wide = build_wide_per_cycle(dva_std, id_col, cyc_col, snap_col, sensors, snapshots,
                                        cfg["missing_snapshot_fill_value"], target_cols=targets)
        feat_cols = wide_feature_columns(sensors, snapshots)
        
        ds_va = WindowDataset(va_wide, feat_cols, targets, id_col, cyc_col,
                              L=cfg["windows"]["L"], stride=cfg["windows"]["stride"], min_cycles=cfg["windows"]["min_cycles"])
        dl_va = DataLoader(ds_va, batch_size=cfg["tcn"]["batch_size"], shuffle=False, num_workers=0)
        
        y_true, y_pred = evaluate_fold(ae, tcn, dl_va, dev)
        
        fold_res = {}
        for i, t in enumerate(targets):
            fold_res[t] = {"MAE": mae(y_true[:, i], y_pred[:, i]),
                           "RMSE": rmse(y_true[:, i], y_pred[:, i])}
            print(f"{t}: MAE={fold_res[t]['MAE']:.2f} RMSE={fold_res[t]['RMSE']:.2f}")
        
        all_scores.append(fold_res)
    
    print("\n" + "="*60)
    print("  LOEO FINAL SCORES")
    print("="*60)
    for t in targets:
        maes = [s[t]["MAE"] for s in all_scores]
        rmses = [s[t]["RMSE"] for s in all_scores]
        print(f"{t}: MAE={np.mean(maes):.2f}±{np.std(maes):.2f} RMSE={np.mean(rmses):.2f}±{np.std(rmses):.2f}")

if __name__ == "__main__":
    main()
