import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import load_config

def generate_top_features(df, snapshot=4):
    """
    Genera solo le top-10 feature per ogni target.
    """
    d = df[df['Snapshot'] == snapshot].copy()
    
    # Top features da forward selection
    # HPT (migliori)
    d['diff_T3_T5'] = d['Sensed_T3'] - d['Sensed_T5']
    d['ratio_Mach_WFuel'] = d['Sensed_Mach'] / (d['Sensed_WFuel'].abs() + 1e-6)
    d['diff_WFuel_P25'] = d['Sensed_WFuel'] - d['Sensed_P25']
    d['ratio_T25_T45'] = d['Sensed_T25'] / (d['Sensed_T45'] + 1e-6)
    d['ratio_T3_T45'] = d['Sensed_T3'] / (d['Sensed_T45'] + 1e-6)
    d['diff_Fan_Core_Speed'] = d['Sensed_Fan_Speed'] - d['Sensed_Core_Speed']
    
    # HPC (top MI)
    d['ratio_T3_T5'] = d['Sensed_T3'] / (d['Sensed_T5'] + 1e-6)
    d['ratio_TAT_T5'] = d['Sensed_TAT'] / (d['Sensed_T5'] + 1e-6)
    d['ratio_T25_T5'] = d['Sensed_T25'] / (d['Sensed_T5'] + 1e-6)
    
    # WW (anche se MI bassa)
    d['ratio_Core_Speed_T5'] = d['Sensed_Core_Speed'] / (d['Sensed_T5'] + 1e-6)
    d['ratio_Ps3_P25'] = d['Sensed_Ps3'] / (d['Sensed_P25'] + 1e-6)
    
    return d

def plot_feature_vs_rul(df, feature_col, target_col, esn, out_dir):
    """
    Plot feature vs RUL per singolo ESN, con eventi manutenzione.
    """
    d = df[df['ESN'] == esn].copy()
    
    # Ordina per ciclo
    d = d.sort_values('Cycles')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Feature vs ciclo
    ax1.plot(d['Cycles'], d[feature_col], 'b-', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel(feature_col, fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'ESN {esn} - {feature_col} vs Cycles', fontsize=12, fontweight='bold')
    
    # Aggiungi eventi manutenzione
    maintenance_events = {
        'Cycles_to_WW': ('WW', 'red'),
        'Cycles_to_HPC_SV': ('HPC', 'orange'),
        'Cycles_to_HPT_SV': ('HPT', 'purple')
    }
    
    for target, (label, color) in maintenance_events.items():
        # Trova cicli dove target = 0 (evento manutenzione)
        events = d[d[target] == 0]['Cycles'].values
        for event_cycle in events:
            ax1.axvline(event_cycle, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
            ax2.axvline(event_cycle, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Plot 2: RUL vs ciclo
    ax2.plot(d['Cycles'], d[target_col], 'g-', alpha=0.7, linewidth=1.5)
    ax2.set_ylabel(f'{target_col} (RUL)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Cycles', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # RUL decresce nel tempo
    
    # Legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label='WW event'),
        Line2D([0], [0], color='orange', linestyle='--', label='HPC event'),
        Line2D([0], [0], color='purple', linestyle='--', label='HPT event')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Salva
    safe_fname = feature_col.replace('/', '_').replace(' ', '_')
    plt.savefig(f"{out_dir}/trend_{safe_fname}_ESN{esn}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: trend_{safe_fname}_ESN{esn}.png")

def plot_feature_correlation_with_rul(df, feature_col, target_col, out_dir):
    """
    Scatter plot feature vs RUL per tutti ESN (verifica correlazione).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    esns = sorted(df['ESN'].unique())
    
    for idx, esn in enumerate(esns):
        ax = axes[idx]
        d = df[df['ESN'] == esn].copy()
        
        # Scatter
        ax.scatter(d[feature_col], d[target_col], alpha=0.5, s=10, c=d['Cycles'], 
                   cmap='viridis', edgecolors='none')
        
        # Fit lineare
        from scipy.stats import pearsonr
        mask = ~(np.isnan(d[feature_col]) | np.isnan(d[target_col]))
        if mask.sum() > 10:
            corr, pval = pearsonr(d[feature_col][mask], d[target_col][mask])
            
            # Linear fit
            z = np.polyfit(d[feature_col][mask], d[target_col][mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(d[feature_col].min(), d[feature_col].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, 
                    label=f'Corr={corr:.2f} (p={pval:.1e})')
        
        ax.set_xlabel(feature_col, fontsize=9)
        ax.set_ylabel(target_col, fontsize=9)
        ax.set_title(f'ESN {esn}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{feature_col} vs {target_col} (all ESNs)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    safe_fname = feature_col.replace('/', '_').replace(' ', '_')
    plt.savefig(f"{out_dir}/scatter_{safe_fname}_vs_{target_col}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: scatter_{safe_fname}_vs_{target_col}.png")

def plot_feature_distribution_by_rul_bins(df, feature_col, target_col, out_dir):
    """
    Boxplot feature distribution per bins di RUL (vedi se cambia vicino a manutenzione).
    """
    d = df.copy()
    
    # Crea bins RUL: 0-100, 100-300, 300-600, 600+
    d['RUL_bin'] = pd.cut(d[target_col], bins=[0, 100, 300, 600, 10000], 
                          labels=['0-100', '100-300', '300-600', '600+'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    esns = sorted(df['ESN'].unique())
    
    for idx, esn in enumerate(esns):
        ax = axes[idx]
        d_esn = d[d['ESN'] == esn].copy()
        
        # Boxplot
        d_esn.boxplot(column=feature_col, by='RUL_bin', ax=ax)
        ax.set_title(f'ESN {esn}', fontsize=10, fontweight='bold')
        ax.set_xlabel('RUL bin (cycles to maintenance)', fontsize=9)
        ax.set_ylabel(feature_col, fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{feature_col} distribution by {target_col} bins', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    safe_fname = feature_col.replace('/', '_').replace(' ', '_')
    plt.savefig(f"{out_dir}/boxplot_{safe_fname}_by_{target_col}_bins.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: boxplot_{safe_fname}_by_{target_col}_bins.png")

def main():
    print("=== EDA Top Features ===\n")
    
    # Carica dati
    cfg = load_config("configs/config.yaml")
    df = pd.read_csv(cfg["data"]["train_csv"])
    print(f"Loaded: {df.shape[0]} rows, {df['ESN'].nunique()} ESNs")
    
    # Genera top features
    print("\nGenerating top features (snapshot 4)...")
    df_feat = generate_top_features(df, snapshot=4)
    
    # Aggrega per ciclo
    top_features = [
        'diff_T3_T5', 'ratio_T3_T45', 'ratio_T3_T5',
        'ratio_TAT_T5', 'ratio_T25_T45', 'ratio_Mach_WFuel',
        'diff_Fan_Core_Speed', 'ratio_Ps3_P25'
    ]
    
    agg_dict = {f: 'mean' for f in top_features}
    agg_dict.update({
        'Cycles_to_WW': 'first',
        'Cycles_to_HPC_SV': 'first',
        'Cycles_to_HPT_SV': 'first'
    })
    
    df_agg = df_feat.groupby(['ESN', 'Cycles']).agg(agg_dict).reset_index()
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"After aggregation: {df_agg.shape[0]} rows")
    
    # Output dir
    import os
    out_dir = "artifacts/eda_top_features"
    os.makedirs(out_dir, exist_ok=True)
    
    # Analizza feature principali per ogni target
    feature_target_pairs = [
        # HPT (best)
        ('diff_T3_T5', 'Cycles_to_HPT_SV'),
        ('ratio_T3_T45', 'Cycles_to_HPT_SV'),
        ('ratio_T3_T5', 'Cycles_to_HPT_SV'),
        
        # HPC
        ('diff_T3_T5', 'Cycles_to_HPC_SV'),
        ('ratio_TAT_T5', 'Cycles_to_HPC_SV'),
        ('ratio_T25_T45', 'Cycles_to_HPC_SV'),
        
        # WW (anche se bassa MI)
        ('ratio_Ps3_P25', 'Cycles_to_WW'),
        ('diff_Fan_Core_Speed', 'Cycles_to_WW'),
    ]
    
    for feature, target in feature_target_pairs:
        print(f"\n{'='*60}")
        print(f"Analyzing: {feature} vs {target}")
        print('='*60)
        
        # 1) Trend plots per ESN
        print("Plotting trends...")
        for esn in sorted(df_agg['ESN'].unique()):
            plot_feature_vs_rul(df_agg, feature, target, esn, out_dir)
        
        # 2) Scatter correlazione
        print("Plotting scatter...")
        plot_feature_correlation_with_rul(df_agg, feature, target, out_dir)
        
        # 3) Boxplot per RUL bins
        print("Plotting boxplot by RUL bins...")
        plot_feature_distribution_by_rul_bins(df_agg, feature, target, out_dir)
    
    print(f"\n=== Tutti i plot salvati in {out_dir}/ ===")
    print("\nFile pattern:")
    print("  - trend_<feature>_ESN<N>.png : Feature + RUL nel tempo con eventi manutenzione")
    print("  - scatter_<feature>_vs_<target>.png : Correlazione feature↔RUL per tutti ESN")
    print("  - boxplot_<feature>_by_<target>_bins.png : Distribuzione feature per RUL bins")

if __name__ == "__main__":
    main()
