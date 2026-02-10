import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils import load_config
from src.io_paths import ensure_dir
from src.snapshot_scaler import SnapshotStandardScaler
from src.wide_builder import build_wide_per_cycle, wide_feature_columns

def compute_vif(X):
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif.sort_values("VIF", ascending=False)

def main(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    df = pd.read_csv(cfg["data"]["train_csv"])
    
    id_col = cfg["schema"]["id_col"]
    cyc_col = cfg["schema"]["cycle_train_col"]
    snap_col = cfg["schema"]["snapshot_col"]
    sensors = cfg["schema"]["sensors"]
    targets = cfg["targets"]
    snapshots = cfg["snapshots"]

    print("=== Dataset shape ===")
    print(df.shape)
    print("\n=== ESN counts ===")
    print(df[id_col].value_counts())
    
    print("\n=== Targets summary ===")
    print(df[targets].describe())

    # Check missing
    print("\n=== Missing values ===")
    print(df[sensors + targets].isnull().sum())

    # Correlation heatmap (sample)
    sample = df.sample(min(5000, len(df)))
    plt.figure(figsize=(10,8))
    sns.heatmap(sample[sensors].corr(), cmap="coolwarm", center=0)
    plt.title("Sensor correlation (sample)")
    plt.tight_layout()
    out_dir = ensure_dir("artifacts/eda")
    plt.savefig(f"{out_dir}/sensor_corr.png", dpi=120)
    plt.close()
    
    # Wide table for VIF
    scaler = SnapshotStandardScaler(sensors, snap_col, snapshots).fit(df)
    df_std = scaler.transform(df)
    wide = build_wide_per_cycle(df_std, id_col, cyc_col, snap_col, sensors, snapshots, 0.0, target_cols=targets)
    feat_cols = wide_feature_columns(sensors, snapshots)
    
    # VIF on sample
    vif_sample = wide[feat_cols].sample(min(1000, len(wide))).dropna()
    vif_df = compute_vif(vif_sample)
    print("\n=== VIF (top 20) ===")
    print(vif_df.head(20))
    vif_df.to_csv(f"{out_dir}/vif.csv", index=False)
    
    print(f"\nSaved EDA outputs to {out_dir}")

if __name__ == "__main__":
    main()
