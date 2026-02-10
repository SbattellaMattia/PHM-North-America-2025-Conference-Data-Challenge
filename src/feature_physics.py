import numpy as np
import pandas as pd
from src.utils import load_config

def add_physics_features(df, snapshot=4):
    """
    Calcola feature termodinamiche su snapshot specifico (default 4 = max load).
    """
    # Filtra snapshot
    d = df[df['Snapshot'] == snapshot].copy()
    
    gamma = 1.4  # specific heat ratio
    
    # 1) Isentropic Efficiency (HPC compressor)
    # Serve P25 (non disponibile diretto) → stimiamo con Pt2 o Pamb
    # Approssimazione: P25 ≈ Pt2 (inlet total pressure)
    d['P25_approx'] = d['Sensed_Pt2']
    
    numerator = d['Sensed_T25'] * (
        np.power(d['Sensed_Ps3'] / d['P25_approx'], (gamma - 1) / gamma) - 1
    )
    denominator = d['Sensed_T3'] - d['Sensed_T25']
    d['isentropic_eff'] = numerator / (denominator + 1e-6)
    
    # 2) Increment Rate T3 → T45 (HPT turbine)
    d['increment_T3_T45'] = (d['Sensed_T45'] - d['Sensed_T3']) / (d['Sensed_T3'] + 1e-6)
    
    # 3) Pressure Ratio (compressor)
    d['pressure_ratio'] = d['Sensed_Ps3'] / (d['Sensed_Pamb'] + 1e-6)
    
    # 4) EGT margin proxy (exhaust gas temperature margin)
    d['EGT_margin_proxy'] = d['Sensed_T45'] - d['Sensed_T3']
    
    # 5) Fuel efficiency
    d['fuel_eff'] = d['Sensed_WFuel'] / (d['Sensed_Fan_Speed'] + 1e-6)
    
    # 6) Thrust proxy
    d['thrust_proxy'] = d['Sensed_VAFN'] * d['Sensed_Fan_Speed']
    
    # 7) Core speed / Fan speed ratio
    d['speed_ratio'] = d['Sensed_Core_Speed'] / (d['Sensed_Fan_Speed'] + 1e-6)
    
    return d

def main(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    df = pd.read_csv(cfg["data"]["train_csv"])
    
    # Aggiungi feature su snapshot 4 (max load)
    df_phys = add_physics_features(df, snapshot=4)
    
    # Correlazione con target
    phys_features = ['isentropic_eff', 'increment_T3_T45', 'pressure_ratio',
                     'EGT_margin_proxy', 'fuel_eff', 'thrust_proxy', 'speed_ratio']
    targets = cfg["targets"]
    
    corr = df_phys[phys_features + targets].corr()
    corr_phys_targets = corr.loc[phys_features, targets]
    
    print("=== Correlazione Physics Features ↔ Targets ===")
    print(corr_phys_targets.to_string())
    
    # Salva
    import os
    from src.io_paths import ensure_dir
    out_dir = ensure_dir("artifacts/eda")
    corr_phys_targets.to_csv(f"{out_dir}/corr_physics_targets.csv")
    
    # Plot heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_phys_targets, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Physics Features ↔ Targets")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/corr_physics_heatmap.png", dpi=150)
    print(f"Saved: {out_dir}/corr_physics_heatmap.png")

if __name__ == "__main__":
    main()
