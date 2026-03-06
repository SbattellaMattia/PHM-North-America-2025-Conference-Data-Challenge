import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import math
import seaborn as sns
import os
from src.utils import load_config
from sklearn.model_selection import GridSearchCV, GroupKFold

# ==========================================
# 0.Visualizzazione e salvataggio
# ==========================================

def save_results(fold_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = [{
        'left_out_esn': fr['left_out_esn'],
        'mae':          fr['mae'],
        'r2':           fr['r2'],
    } for fr in fold_results]

    df_res = pd.DataFrame(rows)
    path   = os.path.join(out_dir, 'HPC_Prediction.csv')
    df_res.to_csv(path, index=False)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(df_res.to_string(index=False))
    print(f"\n  Avg MAE : {df_res['mae'].mean():.1f}")
    print(f"  Avg R²  : {df_res['r2'].mean():.3f}")
    print("="*70)
    return df_res

def plot_results(fold_results, out_dir):
    """
    Genera i grafici per i risultati della validazione incrociata.
    
    fold_results: lista di dizionari, dove ogni dizionario contiene:
                  'left_out_esn', 'info_test', 'y_true', 'y_pred', 
                  'mae', 'twe', 'r2', 'improvement'
    out_dir: cartella dove salvare i grafici
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Grafici per singolo fold
    for fr in fold_results:
        esn    = fr['left_out_esn']
        cycles = fr['info_test']['Cycles'].values
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(cycles, fr['y_true'], '-o', ms=3, lw=2, alpha=0.85,
                label='Actual', color='#2E86AB')
        ax.plot(cycles, fr['y_pred'], '--', lw=2.5, alpha=0.80,
                label='Predicted', color='#F18F01')
        ax.set_xlabel('Cycles', fontweight='bold')
        ax.set_ylabel('RUL to HPC (cycles)', fontweight='bold')
        ax.set_title(f"Hybrid HPC – ESN {esn} | "
                     f"MAE={fr['mae']:.1f}, R²={fr['r2']:.3f}",
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'loeo_esn{esn}.png'), dpi=150)
        plt.close()

    # 2. Summary Panel
    esns  = [fr['left_out_esn'] for fr in fold_results]
    maes  = [fr['mae']          for fr in fold_results]
    r2s   = [fr['r2']           for fr in fold_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Bar Charts
    for ax, vals, title, ylabel, fmt in [
        (axes[0,0], maes, 'MAE per Engine',    'MAE (cycles)', '{:.1f}'),
        (axes[0,1], r2s, 'R² Score',     'R²',         '{:.4f}'),
    ]:
        bars = ax.bar(range(len(esns)), vals, edgecolor='black')
        ax.set_xticks(range(len(esns)))
        ax.set_xticklabels([f'ESN {e}' for e in esns], fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    fmt.format(v), ha='center', va='bottom', fontweight='bold')

    # Scatter Plot
    ax = axes[1, 0]
    colors = ['#e74c3c','#3498db','#2ecc71','#f39c12']
    for i, fr in enumerate(fold_results):
        ax.scatter(fr['y_true'], fr['y_pred'], s=15, alpha=0.45,
                   color=colors[i%4], label=f"ESN {fr['left_out_esn']}")
    all_y = np.concatenate([fr['y_true'] for fr in fold_results])
    lim   = [0, all_y.max()*1.05]
    ax.plot(lim, lim, 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual RUL', fontweight='bold')
    ax.set_ylabel('Predicted RUL', fontweight='bold')
    ax.set_title('Actual vs Predicted', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Text Summary
    ax = axes[1, 1]
    ax.axis('off')
    txt = (f"HPC RESULTS\n{'='*38}\n\n"
           f"Avg MAE:  {np.mean(maes):.1f} cycles\n"
           f"Avg R²:   {np.mean(r2s):.3f}\n"
           "Per-fold:\n")
    for fr in fold_results:
        txt += (f"  ESN {fr['left_out_esn']}: "
                f"MAE={fr['mae']:.1f}, "
                f"R²={fr['r2']:.3f}\n")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.7))

    plt.suptitle("HPC Prediction",
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(os.path.join(out_dir, "HPC"), dpi=150)
    plt.close()
    print(f"\n  ✓ Plots saved → {out_dir}/")

# ==========================================
# 1. Caricamento Dati
# ==========================================
print("\nLoading data...")
cfg = load_config("configs/config.yaml")
df = pd.read_csv(cfg["data"]["train_clean_csv"])
df = df.rename(columns={'Cycles_Since_New': 'Cycles'})
df = df.dropna().reset_index(drop=True)

# ==========================================
# 2. Definizione dei Gruppi di Sensori
# ==========================================
operating_sensors = [
    "Sensed_Mach", "Sensed_Altitude", "Sensed_Pamb", "Sensed_TAT",
    "Sensed_VAFN", "Sensed_VBV", "Sensed_Fan_Speed", "Sensed_Pt2"
]

degradation_sensors = [
    col for col in df.columns 
    if col.startswith('Sensed_') and col not in operating_sensors
] + [
    "Cumulative_HPC_SVs"
]

print(degradation_sensors)

# ==========================================
# 3. Calcolo dei Residui (Engine by Engine)
# ==========================================
engines_data = []

for esn in df['ESN'].unique():
    df_engine = df[df['ESN'] == esn].copy()
    X_op = df_engine[operating_sensors]
    
    for target_sensor in degradation_sensors:
        y_deg = df_engine[target_sensor]
        lr = LinearRegression()
        lr.fit(X_op, y_deg)
        y_pred = lr.predict(X_op)
        
        df_engine[f"{target_sensor}_res"] = y_deg - y_pred
        
    engines_data.append(df_engine)

df_residuals = pd.concat(engines_data).reset_index(drop=True)

# ==========================================
# 4. Raggruppamento per Ciclo
# ==========================================
target_cols = ['Cycles_to_HPC_SV']
residual_cols = [f"{col}_res" for col in degradation_sensors]
others_cols = ['HPC_Eff_Index_clean']

df_agg = df_residuals.groupby(['ESN', 'Cycles'])[target_cols + residual_cols + others_cols].median().reset_index()

# ==========================================
# 5. PREPARAZIONE FEATURE PER LIGHTGBM
# ==========================================
residual_features = [col for col in df_agg.columns if col.endswith('_res')]

for col in residual_features:
    # Trend 50 cicli
    df_agg[f"{col}_roll50"] = df_agg.groupby("ESN")[col].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
    df_agg[f"{col}_diff50"] = df_agg.groupby("ESN")[col].diff(50)

df_agg = df_agg.dropna().reset_index(drop=True)

# Lista feature finale 
feature_cols = (
    residual_features +
    [f"{col}_roll50" for col in residual_features] + 
    [f"{col}_diff50" for col in residual_features] +
    others_cols
)

#print(f"\nFeature che passeremo al modello ({len(feature_cols)}):", feature_cols)


# ==========================================
# 6. HYPERPARAMETER TUNING 
# ==========================================
print("\n--- INIZIO RICERCA MIGLIORI PARAMETRI (GRID SEARCH) ---")
X_all = df_agg[feature_cols]
y_all = df_agg["Cycles_to_HPC_SV"]
groups = df_agg["ESN"] 

# Griglia basata sui tuoi parametri originali
param_grid = {
    'n_estimators': [50, 100, 500],
    'learning_rate': [0.01, 0.02, 0.05],
    'num_leaves': [31, 64],
    'subsample': [0.8],
    'colsample_bytree': [0.6]
}

base_model = LGBMRegressor(max_depth=-1, subsample_freq=1, random_state=42)
gkf = GroupKFold(n_splits=3) # Divide per motore

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=gkf,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_all, y_all, groups=groups)
best_params = grid_search.best_params_

print(f"\n=> MIGLIORI PARAMETRI TROVATI: {best_params}")
print(f"=> MIGLIOR MAE (in validazione): {-grid_search.best_score_:.2f}\n")

# ==========================================
# 6. Training Loop (LOEO)
# ==========================================
engines = df_agg["ESN"].unique()
all_results = []
metrics_per_engine = {}
fold_results = []

for test_engine in engines:
    print(f"--- Elaborazione Motore {test_engine} ---")
    
    # Usiamo df_agg
    df_train = df_agg[df_agg["ESN"] != test_engine]
    df_test = df_agg[df_agg["ESN"] == test_engine].copy()

    X_train = df_train[feature_cols]
    y_train = df_train["Cycles_to_HPC_SV"]
    X_test = df_test[feature_cols]

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.6, 
        random_state=42
    )

    model.fit(X_train, y_train)

    df_test["RUL_pred"] = model.predict(X_test).clip(min=0)
  
    
    # Calcolo metriche
    mae_engine = mean_absolute_error(df_test["Cycles_to_HPC_SV"], df_test["RUL_pred"])
    r2_engine = r2_score(df_test["Cycles_to_HPC_SV"], df_test["RUL_pred"])
    metrics_per_engine[test_engine] = {"MAE": mae_engine, "R2": r2_engine}
    
    print(f"MAE: {mae_engine:.2f} | R²: {r2_engine:.3f}")
    all_results.append(df_test)


    fold_results.append({
    "left_out_esn": test_engine,
    "y_true": df_test["Cycles_to_HPC_SV"].values,
    "y_pred": df_test["RUL_pred"].values,
    "mae": mae_engine,
    "r2": r2_engine,
    "twe": np.mean(np.abs(df_test["Cycles_to_HPC_SV"] - df_test["RUL_pred"])),  # puoi cambiare formula
    "improvement": 0,  # placeholder se non hai baseline
    "info_test": df_test[["Cycles"]]

})
    # Esempio di chiamata
plot_results(fold_results, "outputHPC")
save_results(fold_results, "outputHPC")

# ==========================================
# 7. Ulteriore Visualizzazione
# ==========================================
df_all = pd.concat(all_results)
rows = math.ceil(len(engines) / 2)

fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, esn in enumerate(engines):
    data = df_all[df_all["ESN"] == esn]
    mae = metrics_per_engine[esn]["MAE"]
    r2 = metrics_per_engine[esn]["R2"]
    
    axes[i].plot(data["Cycles"], data["Cycles_to_HPC_SV"], label="Reale", color="black", linewidth=2)
    axes[i].plot(data["Cycles"], data["RUL_pred"], label="Predetto", color="red", linestyle="--")
    axes[i].set_title(f"Motore {esn} | MAE: {mae:.1f} | R²: {r2:.2f}")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join("outputHPC", "LGBMRegressor.png"))

