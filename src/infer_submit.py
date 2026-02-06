import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import load_config, set_seed, device, list_files_from_glob, basename_no_ext
from snapshot_scaler import SnapshotStandardScaler
from wide_builder import build_wide_per_cycle, wide_feature_columns
from models_ae import TimeStepAE
from models_tcn import TCNRegressor

@torch.no_grad()
def predict_file(df_long, cfg, scaler, ae, tcn):
    id_col = cfg["schema"]["id_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    cyc_col = cfg["schema"]["cycle_eval_col"]  # val/test use "Cycles"
    sensors = cfg["schema"]["sensors"]
    snapshots = cfg["snapshots"]

    # standardize
    df_std = scaler.transform(df_long)

    # wide per cycle
    wide = build_wide_per_cycle(df_std, id_col, cyc_col, snap_col, sensors, snapshots, cfg["missing_snapshot_fill_value"])
    feat_cols = wide_feature_columns(sensors, snapshots)

    # take last L cycles (pad if needed)
    L = cfg["windows"]["L"]
    X = wide[feat_cols].astype(np.float32).to_numpy()
    if len(X) >= L:
        Xw = X[-L:]
    else:
        pad = np.zeros((L - len(X), X.shape[1]), dtype=np.float32)
        Xw = np.vstack([pad, X])

    Xw = torch.from_numpy(Xw).unsqueeze(0).to(device())  # [1,L,D]
    _, z = ae(Xw)
    yhat = tcn(z).cpu().numpy().reshape(-1)  # [3]
    return yhat

def main(cfg_path="configs/config.yaml", split="test"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    dev = device()

    # load scaler + models
    scaler = SnapshotStandardScaler.load(os.path.join(cfg["data"]["artifacts_dir"], "snapshot_scaler.json"))

    ae_ckpt = torch.load(os.path.join(cfg["data"]["models_dir"], "ae.pt"), map_location="cpu")
    ae = TimeStepAE(ae_ckpt["in_dim"], cfg["ae"]["latent_dim"], cfg["ae"]["hidden_dim"], cfg["ae"]["dropout"])
    ae.load_state_dict(ae_ckpt["state_dict"])
    ae = ae.to(dev).eval()

    tcn_ckpt = torch.load(os.path.join(cfg["data"]["models_dir"], "tcn.pt"), map_location="cpu")
    tcn = TCNRegressor(in_dim=cfg["ae"]["latent_dim"],
                       channels=tuple(cfg["tcn"]["channels"]),
                       kernel_size=cfg["tcn"]["kernel_size"],
                       dropout=cfg["tcn"]["dropout"],
                       out_dim=3)
    tcn.load_state_dict(tcn_ckpt["state_dict"])
    tcn = tcn.to(dev).eval()

    # file list
    glob_pat = cfg["data"]["test_glob"] if split == "test" else cfg["data"]["val_glob"]
    files = list_files_from_glob(glob_pat)

    # template
    sub = pd.read_csv(cfg["data"]["submission_template"]).copy()
    # Ensure expected columns
    exp_cols = ["file", "Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
    for c in exp_cols:
        if c not in sub.columns:
            raise ValueError(f"submission template missing column: {c}")

    preds = {}
    for fp in tqdm(files, desc=f"Predict {split}"):
        df = pd.read_csv(fp)
        # keep only expected columns
        keep = [cfg["schema"]["id_col"], cfg["schema"]["cycle_eval_col"], cfg["schema"]["snapshot_col"]] + cfg["schema"]["sensors"]
        df = df[keep].copy()

        key = basename_no_ext(fp)  # "test_0"
        yhat = predict_file(df, cfg, scaler, ae, tcn)
        yhat = np.clip(yhat, cfg["inference"]["clip_min"], None)
        preds[key] = yhat

    # fill submission
    for i in range(len(sub)):
        f = str(sub.loc[i, "file"])
        if f not in preds:
            raise ValueError(f"Missing prediction for file={f}. Found keys like: {list(preds)[:5]}")
        sub.loc[i, "Cycles_to_WW"] = float(preds[f][0])
        sub.loc[i, "Cycles_to_HPC_SV"] = float(preds[f][1])
        sub.loc[i, "Cycles_to_HPT_SV"] = float(preds[f][2])

    out = os.path.join(cfg["data"]["artifacts_dir"], f"submission_{split}.csv")
    sub.to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    main()
