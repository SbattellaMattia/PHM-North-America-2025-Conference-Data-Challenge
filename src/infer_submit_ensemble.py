import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils import load_config, set_seed, device, list_files_from_glob, basename_no_ext
from src.snapshot_scaler import SnapshotStandardScaler
from src.wide_builder import build_wide_per_cycle, wide_feature_columns
from src.models_ae import TimeStepAE
from src.models_tcn import TCNRegressor

@torch.no_grad()
def predict_file_with_fold_model(df_long, cfg, scaler, ae, tcn, dev):
    id_col = cfg["schema"]["id_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    cyc_col = cfg["schema"]["cycle_eval_col"]
    sensors = cfg["schema"]["sensors"]
    snapshots = cfg["snapshots"]
    
    df_std = scaler.transform(df_long)
    wide = build_wide_per_cycle(df_std, id_col, cyc_col, snap_col, sensors, snapshots, 
                                 cfg["missing_snapshot_fill_value"])
    feat_cols = wide_feature_columns(sensors, snapshots)
    
    L = cfg["windows"]["L"]
    X = wide[feat_cols].astype(np.float32).to_numpy()
    if len(X) >= L:
        Xw = X[-L:]
    else:
        pad = np.zeros((L - len(X), X.shape[1]), dtype=np.float32)
        Xw = np.vstack([pad, X])
    
    Xw = torch.from_numpy(Xw).unsqueeze(0).to(dev)
    _, z = ae(Xw)
    yhat = tcn(z).cpu().numpy().reshape(-1)
    return yhat

def main(cfg_path="configs/config.yaml", split="test"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    dev = device()
    
    models_dir = cfg["data"]["models_dir"]
    n_folds = 4
    
    # Load all fold models
    fold_models = []
    for fold_idx in range(1, n_folds + 1):
        ae_ckpt = torch.load(os.path.join(models_dir, f"ae_fold{fold_idx}.pt"), map_location="cpu", weights_only=False)
        tcn_ckpt = torch.load(os.path.join(models_dir, f"tcn_fold{fold_idx}.pt"), map_location="cpu", weights_only=False)
        
        ae = TimeStepAE(ae_ckpt["in_dim"], cfg["ae"]["latent_dim"], cfg["ae"]["hidden_dim"], cfg["ae"]["dropout"])
        ae.load_state_dict(ae_ckpt["state_dict"])
        ae = ae.to(dev).eval()
        
        tcn = TCNRegressor(in_dim=cfg["ae"]["latent_dim"],
                           channels=tuple(cfg["tcn"]["channels"]),
                           kernel_size=cfg["tcn"]["kernel_size"],
                           dropout=cfg["tcn"]["dropout"],
                           out_dim=3)
        tcn.load_state_dict(tcn_ckpt["state_dict"])
        tcn = tcn.to(dev).eval()
        
        fold_models.append((ae, tcn))
    
    print(f"Loaded {n_folds} fold models (ensemble)")
    
    # Load scaler (from fold 1, or retrain on all train - here use fold1 for simplicity)
    # Better: refit scaler on all train
    df_train = pd.read_csv(cfg["data"]["train_csv"])
    id_col = cfg["schema"]["id_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    sensors = cfg["schema"]["sensors"]
    snapshots = cfg["snapshots"]
    scaler = SnapshotStandardScaler(sensors, snap_col, snapshots).fit(df_train)
    
    # Files
    glob_pat = cfg["data"]["test_glob"] if split == "test" else cfg["data"]["val_glob"]
    files = list_files_from_glob(glob_pat)
    
    # Template
    sub = pd.read_csv(cfg["data"]["submission_template"]).copy()
    
    preds = {}
    for fp in tqdm(files, desc=f"Predict {split} (ensemble)"):
        df = pd.read_csv(fp)
        keep = [id_col, cfg["schema"]["cycle_eval_col"], snap_col] + sensors
        df = df[keep].copy()
        
        key = basename_no_ext(fp)
        
        # Ensemble: average over folds
        fold_preds = []
        for ae, tcn in fold_models:
            yhat = predict_file_with_fold_model(df, cfg, scaler, ae, tcn, dev)
            fold_preds.append(yhat)
        
        yhat_mean = np.mean(fold_preds, axis=0)
        yhat_mean = np.clip(yhat_mean, cfg["inference"]["clip_min"], None)
        preds[key] = yhat_mean
    
    # Fill submission
    for i in range(len(sub)):
        f = str(sub.loc[i, "file"])
        if f not in preds:
            raise ValueError(f"Missing prediction for file={f}")
        sub.loc[i, "Cycles_to_WW"] = float(preds[f][0])
        sub.loc[i, "Cycles_to_HPC_SV"] = float(preds[f][1])
        sub.loc[i, "Cycles_to_HPT_SV"] = float(preds[f][2])
    
    out = os.path.join(cfg["data"]["artifacts_dir"], f"submission_{split}_ensemble.csv")
    sub.to_csv(out, index=False)
    print(f"Saved ensemble submission: {out}")

if __name__ == "__main__":
    main()
