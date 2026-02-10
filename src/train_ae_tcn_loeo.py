import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.utils import load_config, set_seed, device
from src.io_paths import ensure_dir
from src.splits import loeo_folds
from src.snapshot_scaler import SnapshotStandardScaler
from src.wide_builder import build_wide_per_cycle, wide_feature_columns
from src.dataset_windows import WindowDataset
from src.models_ae import TimeStepAE
from src.models_tcn import TCNRegressor
from src.eval_metrics import mae

def loss_fn(name, delta):
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    return nn.HuberLoss(delta=delta)

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
    
    models_dir = ensure_dir(cfg["data"]["models_dir"])
    
    df = pd.read_csv(cfg["data"]["train_csv"])
    id_col = cfg["schema"]["id_col"]
    cyc_col = cfg["schema"]["cycle_train_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    sensors = cfg["schema"]["sensors"]
    targets = cfg["targets"]
    snapshots = cfg["snapshots"]
    
    df = df[[id_col, cyc_col, snap_col] + sensors + targets].copy()
    
    folds = list(loeo_folds(df, id_col))
    print(f"LOEO: {len(folds)} folds")
    
    all_val_scores = []
    
    for fold_idx, (train_esns, val_esns) in enumerate(folds, 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}/{len(folds)}: train={train_esns} val={val_esns}")
        print('='*60)
        
        dtr = df[df[id_col].isin(train_esns)].copy()
        dva = df[df[id_col].isin(val_esns)].copy()
        
        # Fit scaler on train fold only
        scaler = SnapshotStandardScaler(sensors, snap_col, snapshots).fit(dtr)
        dtr_std = scaler.transform(dtr)
        dva_std = scaler.transform(dva)
        
        tr_wide = build_wide_per_cycle(dtr_std, id_col, cyc_col, snap_col, sensors, snapshots, 
                                        cfg["missing_snapshot_fill_value"], target_cols=targets)
        va_wide = build_wide_per_cycle(dva_std, id_col, cyc_col, snap_col, sensors, snapshots,
                                        cfg["missing_snapshot_fill_value"], target_cols=targets)
        
        feat_cols = wide_feature_columns(sensors, snapshots)
        
        ds_tr = WindowDataset(tr_wide, feat_cols, targets, id_col, cyc_col,
                              L=cfg["windows"]["L"], stride=cfg["windows"]["stride"], 
                              min_cycles=cfg["windows"]["min_cycles"])
        ds_va = WindowDataset(va_wide, feat_cols, targets, id_col, cyc_col,
                              L=cfg["windows"]["L"], stride=cfg["windows"]["stride"],
                              min_cycles=cfg["windows"]["min_cycles"])
        
        dl_tr = DataLoader(ds_tr, batch_size=cfg["ae"]["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
        dl_va = DataLoader(ds_va, batch_size=cfg["tcn"]["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
        
        in_dim = len(feat_cols)
        
        # ========== TRAIN AE ==========
        print(f"\n[Fold {fold_idx}] Training AE...")
        ae = TimeStepAE(in_dim, cfg["ae"]["latent_dim"], cfg["ae"]["hidden_dim"], cfg["ae"]["dropout"]).to(dev)
        opt_ae = torch.optim.AdamW(ae.parameters(), lr=cfg["ae"]["lr"])
        crit_ae = nn.SmoothL1Loss()
        
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
            if (ep+1) % 5 == 0 or ep == 0:
                print(f"  AE epoch {ep+1}/{cfg['ae']['epochs']}: loss={sum(losses)/len(losses):.6f}")
        
        # Freeze AE
        for p in ae.parameters():
            p.requires_grad = False
        
        # ========== TRAIN TCN ==========
        print(f"\n[Fold {fold_idx}] Training TCN...")
        tcn = TCNRegressor(in_dim=cfg["ae"]["latent_dim"],
                           channels=tuple(cfg["tcn"]["channels"]),
                           kernel_size=cfg["tcn"]["kernel_size"],
                           dropout=cfg["tcn"]["dropout"],
                           out_dim=len(targets)).to(dev)
        opt_tcn = torch.optim.AdamW(tcn.parameters(), lr=cfg["tcn"]["lr"])
        crit = loss_fn(cfg["tcn"]["loss"], cfg["tcn"]["huber_delta"])
        
        best_val_mae = float('inf')
        patience_counter = 0
        patience = 15
        
        for ep in range(cfg["tcn"]["epochs"]):
            tcn.train()
            tr_losses = []
            for X, y in dl_tr:
                X, y = X.to(dev), y.to(dev)
                with torch.no_grad():
                    _, z = ae(X)
                opt_tcn.zero_grad()
                pred = tcn(z)
                loss = crit(pred, y)
                loss.backward()
                opt_tcn.step()
                tr_losses.append(loss.item())
            
            # Validation
            y_true, y_pred = evaluate_fold(ae, tcn, dl_va, dev)
            val_mae_mean = np.mean([mae(y_true[:, i], y_pred[:, i]) for i in range(len(targets))])
            
            if (ep+1) % 10 == 0 or ep == 0:
                print(f"  TCN epoch {ep+1}/{cfg['tcn']['epochs']}: train_loss={sum(tr_losses)/len(tr_losses):.3f} val_mae={val_mae_mean:.2f}")
            
            # Early stopping
            if val_mae_mean < best_val_mae:
                best_val_mae = val_mae_mean
                patience_counter = 0
                # Save best checkpoint
                torch.save({"state_dict": ae.state_dict(), "in_dim": in_dim, "cfg": cfg["ae"]},
                           os.path.join(models_dir, f"ae_fold{fold_idx}.pt"))
                torch.save({"state_dict": tcn.state_dict(), "cfg": cfg["tcn"]},
                           os.path.join(models_dir, f"tcn_fold{fold_idx}.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {ep+1}")
                    break
        
        # Final validation scores
        y_true, y_pred = evaluate_fold(ae, tcn, dl_va, dev)
        fold_scores = {}
        for i, t in enumerate(targets):
            fold_scores[t] = mae(y_true[:, i], y_pred[:, i])
            print(f"  {t}: MAE={fold_scores[t]:.2f}")
        all_val_scores.append(fold_scores)
    
    # Summary
    print("\n" + "="*60)
    print("  LOEO SUMMARY")
    print("="*60)
    for t in targets:
        maes = [s[t] for s in all_val_scores]
        print(f"{t}: MAE mean={np.mean(maes):.2f} ± {np.std(maes):.2f}")
    
    print(f"\nSaved {len(folds)} fold models in {models_dir}")

if __name__ == "__main__":
    main()
