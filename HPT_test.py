import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import make_scorer

def calculate_engineered_sensors(df):
    """Ricalcola le feature base mancanti nel set di test dai sensori originali."""
    df = df.copy()
    
    # 1. Calcolo del ratio T3 / T45
    if 'ratio_T3_T45' not in df.columns:
        df['ratio_T3_T45'] = df['Sensed_T3'] / df['Sensed_T45'] + 1e-6 
        
    # 2. Calcolo della tri-ratio diff
    if 'tri_ratio_diff_Sensed_Mach_Sensed_T3_Sensed_T45' not in df.columns:
        # Formula dedotta dai tuoi dati di train: (Mach / T45) - (T3 / T45)
        df['tri_ratio_diff_Sensed_Mach_Sensed_T3_Sensed_T45'] = (df['Sensed_Mach'] - df['Sensed_T3'] / df['Sensed_T45']) + 1e-6
        
    return df

# --------------------------------------------------
# 1. FUNZIONI DI PRE-PROCESSING
# --------------------------------------------------

def aggregate_snapshots(df):
    """Aggrega i dati per ogni ciclo di volo."""
    sensors = ['Sensed_T45', 'Sensed_Ps3', 'Sensed_T3', 'ratio_T3_T45',
                'tri_ratio_diff_Sensed_Mach_Sensed_T3_Sensed_T45',
                'Sensed_Core_Speed','Sensed_Fan_Speed', 'Sensed_Mach', 
                'Sensed_P25', 'Sensed_T25', 'Sensed_T5']
    
    agg_dict = {s: ['mean', 'max', 'min'] for s in sensors}
    
    # Se la colonna target esiste (solo nel training), la includiamo
    target_col = 'Cycles_to_HPT_SV'
    if target_col in df.columns:
        agg_dict[target_col] = 'first'
    
    df_agg = df.groupby(['ESN', 'Cycles']).agg(agg_dict)
    df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
    return df_agg.reset_index()

def add_engine_features(df, group_col='ESN'):
    """Calcola differenze rispetto alla baseline e trend (slope)."""
    df = df.copy()
    feature_cols = [c for c in df.columns if 'Sensed' in c or 'tri_ratio' in c or 'ratio' in c]

    def rolling_slope(x, window=15):
        return x.rolling(window, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0,
            raw=False
        )
    
    for col in feature_cols:
        # BASELINING
        first_val = df.groupby(group_col)[col].transform('first')
        df[f"{col}_delta"] = df[col] - first_val

        # ROLLING SLOPE
        df[f"{col}_slope"] = df.groupby(group_col)[f"{col}_delta"].transform(
            lambda x: rolling_slope(x)
        )
    return df

# --------------------------------------------------
# 2. METRICA CUSTOM TWE (Time Weighted Error)
# --------------------------------------------------

def time_weighted_error(y_true, y_pred, alpha=0.02):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    error = y_pred - y_true
    # Penalità asimmetrica: le late predictions (sovrastime) pesano il doppio
    weight = np.where(error >= 0, 2 / (1 + alpha * y_true), 1 / (1 + alpha * y_true))
    return np.mean(weight * (error ** 2))

def twe_scorer_func(y_true_log, y_pred_log):
    return -time_weighted_error(np.expm1(y_true_log), np.expm1(y_pred_log))

# --------------------------------------------------
# 3. PIPELINE DI TRAINING E TEST
# --------------------------------------------------

def main():
    # --- CONFIGURAZIONE PERCORSI ---
    # Modifica questi percorsi con i tuoi file reali
    train_path = "/Users/niccolociotti/Desktop/PHM-North-America-2025-Conference-Data-Challenge/data/train/training_data_clean.csv" # Il file con i 4 motori
    test_folder = "/Users/niccolociotti/Desktop/PHM-North-America-2025-Conference-Data-Challenge/data/test_imputed"
    output_csv = "predictions_test_hpt.csv"

    # 1. CARICAMENTO E TRAINING
    print("Fase 1: Preparazione dati di training...")
    df_train_raw = pd.read_csv(train_path)
    df_train = add_engine_features(aggregate_snapshots(df_train_raw))
    
    features = [c for c in df_train.columns if '_slope' in c or '_delta' in c] + ['Cycles']
    target = 'Cycles_to_HPT_SV_first'
    
    X_train = df_train[features]
    y_train = np.log1p(df_train[target])
    groups = df_train['ESN']

    # 2. TUNING IPERPARAMETRI (opzionale, puoi usare quelli trovati nel notebook)
    print("Fase 2: Ricerca parametri ottimali su tutti i 4 motori...")
    param_dist = {
        "learning_rate": [0.01, 0.03, 0.05],
        "max_iter": [500, 800],
        "max_depth": [3, 5, 7],
        "l2_regularization": [1.0, 5.0],
        "min_samples_leaf": [20, 50]
    }
    
    gkf = GroupKFold(n_splits=4)
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(loss="absolute_error", random_state=42),
        param_distributions=param_dist,
        n_iter=10, cv=gkf, scoring=make_scorer(twe_scorer_func), n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train, groups=groups)
    
    # 3. ADDESTRAMENTO FINALE
    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)
    print("Modello finale addestrato con successo.")

   # 4. PREDIZIONE SUI TEST E CREAZIONE SUBMISSION IN ORDINE
    print("Fase 3: Elaborazione dei file di test in ordine...")
    
    # Carichiamo il template della submission per avere l'ordine perfetto
    # (assicurati che il percorso sia corretto)
    submission_path = "/Users/niccolociotti/Desktop/PHM-North-America-2025-Conference-Data-Challenge/data/submission.csv"
    df_submission = pd.read_csv(submission_path)
    
    # Iteriamo direttamente sulle righe del template
    for idx, row in df_submission.iterrows():
        file_name = row['file'] # es. "test_0"
        f_path = os.path.join(test_folder, f"{file_name}.csv")
        
        if not os.path.exists(f_path):
            print(f"Attenzione: il file {f_path} non è stato trovato!")
            continue
            
        df_test_raw = pd.read_csv(f_path)
        
        # 1. Ricostruzione sensori finti (la funzione vista in precedenza)
        df_test_raw = calculate_engineered_sensors(df_test_raw)
        
        # 2. Pre-processing e Feature Engineering
        df_test = add_engine_features(aggregate_snapshots(df_test_raw))
        
        # 3. Predizione su tutti i cicli per quel file
        preds_log = best_model.predict(df_test[features])
        df_test['RUL_Predicted'] = np.expm1(preds_log)
        
        # 4. Estrazione della predizione finale (l'ultimo ciclo registrato) arrotondato e convertito in intero
        final_rul = int(round(df_test['RUL_Predicted'].iloc[-1]))
        
        # 5. Inseriamo la predizione direttamente nella riga corretta del template
        df_submission.at[idx, 'Cycles_to_HPT_SV'] = final_rul

    # 5. EXPORT FINALE
    df_submission.to_csv(output_csv, index=False)
    print(f"\n--- Processo completato ---")
    print(f"File salvato con ordine perfetto in: {output_csv} ({len(df_submission)} righe)")

if __name__ == "__main__":
    main()