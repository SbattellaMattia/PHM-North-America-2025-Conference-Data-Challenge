import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils import load_config, set_seed
from src.io_paths import ensure_dir

def main(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    
    df = pd.read_csv(cfg["data"]["train_csv"])
    sensors = cfg["schema"]["sensors"]
    targets = cfg["targets"]
    
    esn = 101
    sub = df[df['ESN'] == esn].sort_values('Cycles')
    
    # Aggrega per ciclo (media su snapshot)
    per_cycle = sub.groupby('Cycles').agg({
        **{s: 'mean' for s in sensors},
        **{t: 'first' for t in targets}
    }).reset_index()
    
    out_dir = ensure_dir("artifacts/eda")
    
    # 1) Plot tutti i sensori
    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, n_sensors * 2))
    if n_sensors == 1:
        axes = [axes]
    
    for i, s in enumerate(sensors):
        axes[i].plot(per_cycle['Cycles'], per_cycle[s], linewidth=0.8, alpha=0.7)
        axes[i].set_ylabel(s, fontsize=9)
        axes[i].grid(True, alpha=0.3)
        if i == n_sensors - 1:
            axes[i].set_xlabel('Cycles')
    
    plt.suptitle(f'ESN {esn}: All sensors vs Cycles', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/esn{esn}_all_sensors.png", dpi=150)
    print(f"Saved: {out_dir}/esn{esn}_all_sensors.png")
    plt.close()
    
    # 2) Correlazione sensori ↔ target (su tutto il train, non solo ESN 101)
    # aggrega per ciclo su tutto il dataset
    all_per_cycle = df.groupby(['ESN', 'Cycles']).agg({
        **{s: 'mean' for s in sensors},
        **{t: 'first' for t in targets}
    }).reset_index()
    
    corr_matrix = all_per_cycle[sensors + targets].corr()
    corr_sensors_targets = corr_matrix.loc[sensors, targets]
    
    print("\n=== Correlazione sensori ↔ target (train completo) ===")
    print(corr_sensors_targets.to_string())
    corr_sensors_targets.to_csv(f"{out_dir}/corr_sensors_targets.csv")
    
    # Plot heatmap
    import seaborn as sns
    plt.figure(figsize=(8, 10))
    sns.heatmap(corr_sensors_targets, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Pearson correlation'})
    plt.title("Correlation: Sensors (mean per cycle) ↔ Targets")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/corr_heatmap.png", dpi=150)
    print(f"Saved: {out_dir}/corr_heatmap.png")
    plt.close()
    
    # 3) Identifica top sensori per target
    print("\n=== Top 5 sensori per correlazione assoluta con ciascun target ===")
    for t in targets:
        top = corr_sensors_targets[t].abs().sort_values(ascending=False).head(5)
        print(f"\n{t}:")
        print(top.to_string())

if __name__ == "__main__":
    main()
