import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from src.utils import load_config
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def generate_engineered_features(df, snapshot=4):
    """
    Genera feature candidate (rapporti, differenze, trasformazioni).
    """
    d = df[df['Snapshot'] == snapshot].copy()
    
    sensor_cols = [c for c in d.columns if c.startswith('Sensed_')]
    
    print(f"Sensori trovati: {len(sensor_cols)}")
    
    # 1) Rapporti e differenze
    count = 0
    for i, s1 in enumerate(sensor_cols):
        for s2 in sensor_cols[i+1:]:
            d[f'ratio_{s1}_{s2}'] = d[s1] / (d[s2].abs() + 1e-6)
            d[f'diff_{s1}_{s2}'] = d[s1] - d[s2]
            count += 2
    
    print(f"Feature rapporti/diff generate: {count}")
    
    # 2) Trasformazioni non-lineari
    for s in sensor_cols:
        # Shift per evitare log/sqrt di negativi
        shift = max(0, 1 - d[s].min())
        d[f'log_{s}'] = np.log(d[s] + shift + 1e-6)
        d[f'sqrt_{s}'] = np.sqrt(d[s] + shift)
        d[f'square_{s}'] = d[s] ** 2
    
    print(f"Feature trasformazioni generate: {len(sensor_cols) * 3}")
    
    return d

def mutual_info_ranking(X, y, feature_names, top_k=20):
    """
    Ordina feature per Mutual Information (correlazione non-lineare).
    """
    print("Calcolo Mutual Information...")
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'MI': mi
    }).sort_values('MI', ascending=False)
    return mi_df.head(top_k)

def forward_selection(X, y, feature_names, max_features=10, cv=4):
    """
    Forward feature selection con cross-validation.
    """
    selected_idx = []
    remaining_idx = list(range(X.shape[1]))
    
    print(f"\n{'Step':<5} {'Feature':<50} {'CV MAE':<10}")
    print("="*70)
    
    for step in range(max_features):
        best_score = -np.inf
        best_idx = None
        
        for idx in remaining_idx:
            candidate = selected_idx + [idx]
            X_sub = X[:, candidate]
            
            score = cross_val_score(
                Ridge(alpha=1.0), X_sub, y,
                cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1
            ).mean()
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is None:
            break
        
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)
        
        print(f"{step+1:<5} {feature_names[best_idx]:<50} {-best_score:<10.2f}")
    
    return [feature_names[i] for i in selected_idx]

def main():
    # Carica dati
    print("=== Caricamento dati ===")
    cfg = load_config("configs/config.yaml")
    df = pd.read_csv(cfg["data"]["train_csv"])
    
    targets = ['Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']
    
    print(f"Shape originale: {df.shape}")
    print(f"ESN unici: {df['ESN'].nunique()}")
    
    # Genera feature
    print("\n=== Generazione feature automatiche ===")
    df_eng = generate_engineered_features(df, snapshot=4)
    
    # Escludi colonne non-feature
    exclude_cols = ['ESN', 'Cycles', 'Snapshot', 
                    'Cumulative_WWs', 'Cumulative_HPC_SVs', 'Cumulative_HPT_SVs'] + \
                   targets + \
                   [c for c in df_eng.columns if c.startswith('Commanded_')]
    
    feature_cols = [c for c in df_eng.columns if c not in exclude_cols]
    print(f"Feature totali generate: {len(feature_cols)}")
    
    # Aggrega per ciclo (media snapshot 4, che è già filtrato)
    agg_dict = {f: 'mean' for f in feature_cols}
    agg_dict.update({t: 'first' for t in targets})
    
    df_agg = df_eng.groupby(['ESN', 'Cycles']).agg(agg_dict).reset_index()
    
    print(f"Shape dopo aggregazione: {df_agg.shape}")
    
    # Rimuovi NaN/Inf
    df_clean = df_agg.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Shape dopo pulizia NaN/Inf: {df_clean.shape}")
    
    X = df_clean[feature_cols].values
    
    # Standardizza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"X finale: {X_scaled.shape}")
    
    # Crea cartella output
    import os
    out_dir = "artifacts/feature_discovery"
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    # Per ogni target
    for target in targets:
        print(f"\n{'='*70}")
        print(f"TARGET: {target}")
        print('='*70)
        
        y = df_clean[target].values
        
        # 1) Mutual Information ranking
        print("\n--- Top 20 by Mutual Information ---")
        mi_top = mutual_info_ranking(X_scaled, y, feature_cols, top_k=20)
        print(mi_top.to_string(index=False))
        mi_top.to_csv(f"{out_dir}/MI_top20_{target}.csv", index=False)
        
        # 2) Forward selection
        print("\n--- Forward Selection (CV MAE) ---")
        selected = forward_selection(X_scaled, y, feature_cols, max_features=10, cv=4)
        
        results[target] = {
            'MI_top': mi_top['feature'].tolist(),
            'forward_selected': selected
        }
    
    # Salva summary
    import json
    with open(f"{out_dir}/feature_discovery_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Risultati salvati in {out_dir}/ ===")
    print("File generati:")
    print("  - MI_top20_Cycles_to_WW.csv")
    print("  - MI_top20_Cycles_to_HPC_SV.csv")
    print("  - MI_top20_Cycles_to_HPT_SV.csv")
    print("  - feature_discovery_summary.json")

if __name__ == "__main__":
    main()
