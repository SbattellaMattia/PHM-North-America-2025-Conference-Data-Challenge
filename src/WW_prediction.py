"""
LSTM + Attention per predizione SINGLE-TASK (WW) con LOEO
==========================================================
DATA: 2026-02-18
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from src.utils import load_config


# ============================================================================
# SEZIONE 1: FEATURE ENGINEERING
# ============================================================================

def create_base_features(df):
    print("  Creating engineered features...")
    df = df.copy()
    
    df['temp_diff_norm'] = ((df['Sensed_T45'] - df['Sensed_T25']) / (df['Sensed_T25'] + 1e-6)) ** 2
    df['thermal_stress'] = ((df['Sensed_T45'] - df['Sensed_T25']) * df['Sensed_Core_Speed'] / 1000)
    df['load_factor'] = df['Sensed_WFuel'] * df['Sensed_Core_Speed']
    df['speed_ratio'] = df['Sensed_Core_Speed'] / (df['Sensed_Fan_Speed'] + 1e-6)
    df['temp_gradient'] = ((df['Sensed_T45'] - df['Sensed_T3']) / (df['Sensed_T3'] + 1e-6))
    df['pressure_ratio'] = df['Sensed_Ps3'] / (df['Sensed_TAT'] + 1e-6)
    
    gamma = 1.4
    pressure_ratio_hptc = df['Sensed_Ps3'] / (df['Sensed_P25'] + 1e-6)
    temp_ratio_hptc = df['Sensed_T3'] / (df['Sensed_T25'] + 1e-6)
    df['hptc_efficiency'] = ((pressure_ratio_hptc ** ((gamma - 1) / gamma)) / (temp_ratio_hptc + 1e-6))
    df['hpt_stress_indicator'] = ((df['Sensed_Mach'] - df['Sensed_T3']) / (df['Sensed_T45'] + 1e-6))
    
    print("  Created 8 engineered features")
    return df


def aggregate_by_cycle(df):
    print("  Aggregating 8 snapshots per cycle...")
    exclude_cols = ['ESN', 'Cycles', 'Snapshot', 'Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    agg_dict = {col: 'mean' for col in feature_cols}
    agg_dict['Cycles_to_WW'] = 'first'
    df_agg = df.groupby(['ESN', 'Cycles']).agg(agg_dict).reset_index()
    df_agg = df_agg.sort_values(['ESN', 'Cycles']).reset_index(drop=True)
    print(f"  Before: {len(df):,} → After: {len(df_agg):,} rows")
    return df_agg


def engineer_ww_features(df):
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    print(f"\nInput: {df.shape}, ESNs: {df['ESN'].nunique()}")
    df_feat = create_base_features(df)
    df_agg = aggregate_by_cycle(df_feat)
    df_clean = df_agg.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Final: {df_clean.shape}")
    return df_clean

def diagnose_data(df):
    """Diagnostica completa dei dati per LOEO."""
    print("\n" + "="*80)
    print("🔍 DATA DIAGNOSIS FOR LOEO")
    print("="*80)
    
    for esn in sorted(df['ESN'].unique()):
        df_esn = df[df['ESN'] == esn]
        
        print(f"\n{'─'*80}")
        print(f"ESN {esn}")
        print(f"{'─'*80}")
        print(f"  Total cycles: {len(df_esn)}")
        print(f"  Cycles_to_WW:")
        print(f"    Range: [{df_esn['Cycles_to_WW'].min():.0f}, {df_esn['Cycles_to_WW'].max():.0f}]")
        print(f"    Mean: {df_esn['Cycles_to_WW'].mean():.1f}")
        print(f"    Std: {df_esn['Cycles_to_WW'].std():.1f}")
        print(f"    Median: {df_esn['Cycles_to_WW'].median():.0f}")
        
        # Check feature distributions
        print(f"\n  Top 3 sensors (sample stats):")
        for col in ['Sensed_T45', 'Sensed_Core_Speed', 'Sensed_WFuel']:
            if col in df_esn.columns:
                print(f"    {col}: mean={df_esn[col].mean():.1f}, std={df_esn[col].std():.1f}")
    
    print("\n" + "="*80)
    print("CROSS-ESN COMPARISON")
    print("="*80)
    
    # Compare WW distributions
    ww_stats = []
    for esn in sorted(df['ESN'].unique()):
        ww_stats.append({
            'ESN': esn,
            'WW_mean': df[df['ESN']==esn]['Cycles_to_WW'].mean(),
            'WW_std': df[df['ESN']==esn]['Cycles_to_WW'].std(),
            'WW_max': df[df['ESN']==esn]['Cycles_to_WW'].max(),
            'n_cycles': len(df[df['ESN']==esn])
        })
    
    df_stats = pd.DataFrame(ww_stats)
    print("\n" + df_stats.to_string(index=False))
    
    # Check if ESN 101 is outlier
    mean_ww_overall = df['Cycles_to_WW'].mean()
    for esn in sorted(df['ESN'].unique()):
        esn_mean = df[df['ESN']==esn]['Cycles_to_WW'].mean()
        diff_pct = abs(esn_mean - mean_ww_overall) / mean_ww_overall * 100
        status = "⚠️ OUTLIER" if diff_pct > 30 else "✓ OK"
        print(f"ESN {esn}: {diff_pct:.1f}% from overall mean {status}")
    
    print("\n" + "="*80)

# ============================================================================
# SEZIONE 2: SEQUENCE CREATION (WW ONLY)
# ============================================================================
def create_sequences_ww(df, feature_cols, window_size):
    """Crea sequenze CON informazione di posizione temporale."""
    print(f"\n  Creating sequences (window={window_size})...")
    X_seq, y_ww, cycle_info = [], [], []
    
    for esn in sorted(df['ESN'].unique()):
        d_esn = df[df['ESN'] == esn].sort_values('Cycles').reset_index(drop=True)
        
        # ========== CRITICAL FIX ==========
        # Aggiungi "a che punto della vita siamo"
        max_cycle = d_esn['Cycles'].max()
        d_esn['relative_cycle'] = d_esn['Cycles'] / max_cycle  # [0, 1]
        
        # Esempio:
        # Cycle 500/2000 → relative_cycle = 0.25 (25% della vita)
        # Cycle 1900/2000 → relative_cycle = 0.95 (95% della vita)
        # ==================================
        
        # Aggiungi alle feature
        feature_cols_enhanced = feature_cols + ['relative_cycle']
        
        X = d_esn[feature_cols_enhanced].values
        y = d_esn['Cycles_to_WW'].values
        cycles = d_esn['Cycles'].values
        
        n_sequences = len(X) - window_size + 1
        if n_sequences <= 0:
            continue
        
        for i in range(n_sequences):
            X_seq.append(X[i:i + window_size])
            y_ww.append(y[i + window_size - 1])
            cycle_info.append({'ESN': esn, 'Cycle_end': cycles[i + window_size - 1]})
        
        print(f"    ESN {esn}: {len(X)} cycles → {n_sequences} sequences (23 features)")
    
    X_seq = np.array(X_seq)
    y_ww = np.array(y_ww)
    print(f"\n  Total: {len(X_seq):,} sequences")
    print(f"  Shape: {X_seq.shape} (window={window_size}, features=23)")
    print(f"  WW range: [{y_ww.min():.0f}, {y_ww.max():.0f}]")
    return X_seq, y_ww, pd.DataFrame(cycle_info)


'''def create_sequences_ww(df, feature_cols, window_size):
    print(f"\n  Creating sequences (window={window_size})...")
    X_seq, y_ww, cycle_info = [], [], []
    
    for esn in sorted(df['ESN'].unique()):
        d_esn = df[df['ESN'] == esn].sort_values('Cycles').reset_index(drop=True)
        X = d_esn[feature_cols].values
        y = d_esn['Cycles_to_WW'].values
        cycles = d_esn['Cycles'].values
        n_sequences = len(X) - window_size + 1
        if n_sequences <= 0:
            continue
        for i in range(n_sequences):
            X_seq.append(X[i:i + window_size])
            y_ww.append(y[i + window_size - 1])
            cycle_info.append({'ESN': esn, 'Cycle_end': cycles[i + window_size - 1]})
        print(f"    ESN {esn}: {len(X)} cycles → {n_sequences} sequences")
    
    X_seq = np.array(X_seq)
    y_ww = np.array(y_ww)
    print(f"\n  Total: {len(X_seq):,} sequences, WW range: [{y_ww.min():.0f}, {y_ww.max():.0f}]")
    return X_seq, y_ww, pd.DataFrame(cycle_info)'''


# ============================================================================
# SEZIONE 3: MODEL (SINGLE OUTPUT: WW)
# ============================================================================

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), 
                                initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), 
                                initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)
    def get_config(self):
        return super().get_config()


def build_lstm_attention_ww(timesteps, n_features, lstm_units=[64, 32], dropout=0.3, l2_reg=0.001):
    print(f"\n  Building LSTM+Attention (WW)")
    print(f"    Input: {timesteps} timesteps, {n_features} features")
    
    inputs = keras.Input(shape=(timesteps, n_features), name='input')
    x = inputs
    
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(units, return_sequences=True, 
                       kernel_regularizer=keras.regularizers.l2(l2_reg),
                       recurrent_regularizer=keras.regularizers.l2(l2_reg), 
                       name=f'lstm_{i+1}')(x)
        x = layers.Dropout(dropout)(x)
        print(f"    LSTM {i+1}: {units} units")
    
    x = AttentionLayer(name='attention')(x)
    print(f"    Attention: collapse timesteps")
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation='linear', name='output_ww')(x)
    
    model = Model(inputs=inputs, outputs=output, name='LSTM_Attention_WW')
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    print(f"    Parameters: {model.count_params():,}")
    return model


def build_lstm_attention_custom(timesteps, n_features, dropout=0.2, l2_reg=0.0001):
    """
    ARCHITETTURA SEMPLIFICATA per LOEO.
    
    RIDOTTO:
    - LSTM: 64→32, 32→16 (50% parametri)
    - Dense: 32→16, 16→8 (70% parametri)
    - Dropout: 0.3→0.2 (meno aggressivo)
    - L2: 0.001→0.0001 (meno penalità)
    """
    print(f"\n  Building SIMPLIFIED LSTM+Attention")
    print(f"    Input: {timesteps} timesteps, {n_features} features")
    
    inputs = keras.Input(shape=(timesteps, n_features), name='input')
    x = inputs
    
    # LSTM 1: 32 units (invece di 64)
    x = layers.LSTM(
        32,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        recurrent_regularizer=keras.regularizers.l2(l2_reg),
        name='lstm_1'
    )(x)
    x = layers.Dropout(dropout)(x)
    print(f"    LSTM 1: 32 units")
    
    # LSTM 2: 16 units (invece di 32)
    x = layers.LSTM(
        16,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        recurrent_regularizer=keras.regularizers.l2(l2_reg),
        name='lstm_2'
    )(x)
    x = layers.Dropout(dropout)(x)
    print(f"    LSTM 2: 16 units")
    
    # Attention (uguale)
    x = AttentionLayer(name='attention')(x)
    print(f"    Attention: collapse timesteps")
    
    # Dense 1: 16 units (invece di 32)
    x = layers.Dense(16, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    
    # Dense 2: 8 units (invece di 16)
    x = layers.Dense(8, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    
    # Output
    output = layers.Dense(1, activation='linear', name='output_ww')(x)
    
    model = Model(inputs=inputs, outputs=output, name='LSTM_Attention_WW_Simple')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    total_params = model.count_params()
    print(f"    Parameters: {total_params:,} (vs ~40k originale)")
    
    return model


def weighted_mse_loss(y_true, y_pred):
    """
    MSE loss che penalizza DI PIÙ gli errori agli estremi.
    
    Logica: 
    - Valori vicini a 0 o al max hanno peso 2x
    - Valori centrali hanno peso 1x
    """
    # Calcola distanza dal centro (normalizzato)
    y_mean = 3.5  # Centro di log(1+[0,1150]) ≈ [0, 7]
    distance_from_center = tf.abs(y_true - y_mean) / 3.5  # [0, 1]
    
    # Peso: 1.0 al centro, 2.0 agli estremi
    weights = 1.0 + distance_from_center
    
    # MSE pesato
    squared_diff = tf.square(y_true - y_pred)
    weighted_squared_diff = squared_diff * weights
    
    return tf.reduce_mean(weighted_squared_diff)


def build_lstm_attention_ww_weighted(timesteps, n_features, 
                                     lstm_units=[64, 32], 
                                     dropout=0.2, 
                                     l2_reg=0.0005):
    """LSTM con WEIGHTED LOSS per gli estremi."""
    print(f"\n  Building LSTM+Attention with WEIGHTED LOSS")
    print(f"    Input: {timesteps} timesteps, {n_features} features")
    
    inputs = keras.Input(shape=(timesteps, n_features), name='input')
    x = inputs
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(
            units,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            name=f'lstm_{i+1}'
        )(x)
        x = layers.Dropout(dropout)(x)
        print(f"    LSTM {i+1}: {units} units")
    
    x = AttentionLayer(name='attention')(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation='linear', name='output_ww')(x)
    
    model = Model(inputs=inputs, outputs=output, name='LSTM_Weighted')
    
    # ========== WEIGHTED LOSS! ==========
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_mse_loss,  # ← Custom loss
        metrics=['mae']
    )
    
    print(f"    Loss: Weighted MSE (2x penalty at extremes)")
    print(f"    Parameters: {model.count_params():,}")
    return model


# ============================================================================
# SEZIONE 4: LOEO TRAINING
# ============================================================================
def train_ww_loeo(df, feature_cols, window_size, log_target=True, epochs=300, batch_size=32):
    """
    LOEO training con DEBUG completo per capire cosa sta succedendo.
    """
    esns = sorted(df['ESN'].unique())
    fold_results = []
    
    print("\n" + "="*70)
    print(f"LOEO TRAINING | window={window_size} | engines={esns}")
    print("="*70)
    
    for left_out_esn in esns:
        print(f"\n{'─'*70}\nFOLD: Leave out ESN {left_out_esn}\n{'─'*70}")
        
        # ========== SPLIT PER MOTORE ==========
        df_train = df[df['ESN'] != left_out_esn].copy()
        df_test = df[df['ESN'] == left_out_esn].copy()
        print(f"  Train ESNs: {sorted(df_train['ESN'].unique())}")
        print(f"  Test ESN: [{left_out_esn}]")
        
        # ========== CREATE SEQUENCES ==========
        X_train_seq, y_train, _ = create_sequences_ww(df_train, feature_cols, window_size)
        X_test_seq, y_test, info_test = create_sequences_ww(df_test, feature_cols, window_size)
        
        if len(X_train_seq) < 100 or len(X_test_seq) < 10:
            print("  ⚠️ Too few sequences, skipping")
            continue
        
        print(f"\n  📊 DATA CHECK:")
        print(f"     Train sequences: {len(X_train_seq)}")
        print(f"     Test sequences: {len(X_test_seq)}")
        print(f"     y_train (cycles): [{y_train.min():.0f}, {y_train.max():.0f}], mean={y_train.mean():.1f}")
        print(f"     y_test (cycles): [{y_test.min():.0f}, {y_test.max():.0f}], mean={y_test.mean():.1f}")
        
        # ========== TARGET TRANSFORM (SOLO PER TRAINING!) ==========
        if log_target:
            print("\n  Applying log-transform to target...")
            y_train_log = np.log(y_train + 1)
            y_test_log = np.log(y_test + 1)  # Per validation durante training
            print(f"     y_train_log: [{y_train_log.min():.3f}, {y_train_log.max():.3f}], mean={y_train_log.mean():.3f}")
            print(f"     y_test_log: [{y_test_log.min():.3f}, {y_test_log.max():.3f}], mean={y_test_log.mean():.3f}")
        else:
            y_train_log = y_train
            y_test_log = y_test
        
        # ========== FEATURE SCALING (FIT SOLO SU TRAIN!) ==========
        print("\n  Scaling features (fit on train only)...")
        scaler = StandardScaler()
        X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train_seq.shape)
        
        X_test_flat = X_test_seq.reshape(-1, X_test_seq.shape[-1])
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test_seq.shape)
        
        # ========== SPLIT TRAIN/VAL ==========
        print("\n  Splitting train/validation (80/20)...")
        idx = np.arange(len(X_train_scaled))
        tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
        
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train_log[tr_idx], y_train_log[val_idx]
        print(f"    Train: {len(tr_idx):,}, Val: {len(val_idx):,}")
        
        # ========== BUILD MODEL ==========
        #model = build_lstm_attention_custom(timesteps=window_size, n_features=X_train_seq.shape[-1])
        model = build_lstm_attention_ww(timesteps=window_size, n_features=X_train_seq.shape[-1])
        '''model = build_lstm_attention_ww_simplified(
            timesteps=window_size, 
            n_features=X_train_seq.shape[-1]
        )'''
        
        # ========== TRAINING ==========
        print(f"\n  Training (max {epochs} epochs, batch {batch_size})...")
        t0 = time.time()
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=50,  # Aumentato da 30
                    restore_best_weights=True, 
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=15,  # Aumentato da 10
                    min_lr=1e-6, 
                    verbose=1
                )
            ]
        )
        train_time = time.time() - t0
        print(f"\n  Training completed in {train_time:.1f}s ({len(history.history['loss'])} epochs)")
        
        # ========== PREDICTION SU TEST ENGINE ==========
        print("\n  Evaluating on test engine...")
        t1 = time.time()
        y_pred_log = model.predict(X_test_scaled, verbose=0).flatten()
        inf_time = time.time() - t1
        
        # ========== DEBUG PREDICTIONS ==========
        print(f"\n  🔍 PREDICTION DEBUG:")
        print(f"     Model output (log-scale):")
        print(f"       Range: [{y_pred_log.min():.3f}, {y_pred_log.max():.3f}]")
        print(f"       Mean: {y_pred_log.mean():.3f}")
        print(f"       Std: {y_pred_log.std():.3f}")
        
        # Check se il modello è collassato
        if y_pred_log.std() < 0.05:
            print(f"       ⚠️ WARNING: Model may have collapsed (low std)!")
        
        # ========== INVERSE TRANSFORM ==========
        if log_target:
            print(f"\n     Inverse log-transform:")
            y_pred_cycles = np.clip(np.exp(y_pred_log) - 1, 0, None)
            print(f"       y_pred (cycles): [{y_pred_cycles.min():.1f}, {y_pred_cycles.max():.1f}]")
            print(f"       Mean: {y_pred_cycles.mean():.1f}, Std: {y_pred_cycles.std():.1f}")
        else:
            y_pred_cycles = np.clip(y_pred_log, 0, None)
        
        print(f"\n     Ground truth (cycles):")
        print(f"       y_test: [{y_test.min():.0f}, {y_test.max():.0f}]")
        print(f"       Mean: {y_test.mean():.1f}, Std: {y_test.std():.1f}")
        
        # Sanity check
        mean_diff = abs(y_pred_cycles.mean() - y_test.mean())
        if mean_diff > 300:
            print(f"       ⚠️ WARNING: Large mean difference = {mean_diff:.1f} cycles")
        
        # ========== METRICS (SU CYCLES!) ==========
        mae = mean_absolute_error(y_test, y_pred_cycles)
        r2 = r2_score(y_test, y_pred_cycles)
        corr = np.sqrt(r2) if r2 > 0 else 0.0
        
        # Calcola anche baseline MAE (predire sempre la media)
        baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
        improvement = (baseline_mae - mae) / baseline_mae * 100
        
        print(f"\n  ✅ TEST ESN {left_out_esn}:")
        print(f"     MAE: {mae:.1f} cycles")
        print(f"     R²: {r2:.3f}")
        print(f"     Correlation: {corr:.3f}")
        print(f"     Baseline MAE (mean): {baseline_mae:.1f}")
        print(f"     Improvement: {improvement:+.1f}%")
        print(f"     Train time: {train_time:.1f}s")
        print(f"     Inference time: {inf_time:.3f}s")
        
        # ========== SAVE RESULTS ==========
        fold_results.append({
            'left_out_esn': left_out_esn, 
            'window': window_size, 
            'mae': mae, 
            'r2': r2, 
            'corr': corr,
            'baseline_mae': baseline_mae,
            'improvement_pct': improvement,
            'train_time_s': train_time, 
            'inference_time_s': inf_time, 
            'n_test_seq': len(y_test),
            'y_true': y_test,  # In cycles!
            'y_pred': y_pred_cycles,  # In cycles!
            'info_test': info_test, 
            'history': history.history,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
    
    return fold_results



# ============================================================================
# SEZIONE 5: VISUALIZATION
# ============================================================================

def plot_loeo_predictions(fold_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fr in fold_results:
        esn = fr['left_out_esn']
        cycles = fr['info_test']['Cycle_end'].values
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cycles, fr['y_true'], '-o', markersize=4, linewidth=2.5, 
               alpha=0.9, label='Actual', color='#2E86AB')
        ax.plot(cycles, fr['y_pred'], '--', linewidth=2.8, alpha=0.75, 
               label='Predicted', color='#F18F01')
        ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
        ax.set_ylabel('RUL WW (cycles)', fontsize=12, fontweight='bold')
        ax.set_title(f"LOEO - ESN {esn} | MAE={fr['mae']:.1f}, R²={fr['r2']:.3f}", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        plt.tight_layout()
        
        fname = f"loeo_ww_esn{esn}_window{fr['window']}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fname}")


def plot_loeo_summary(fold_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    esns = [fr['left_out_esn'] for fr in fold_results]
    maes = [fr['mae'] for fr in fold_results]
    r2s = [fr['r2'] for fr in fold_results]
    
    # MAE bar chart
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(esns)), maes, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(esns)))
    ax1.set_xticklabels([f'ESN {e}' for e in esns], fontweight='bold')
    ax1.set_ylabel('MAE (cycles)', fontsize=12, fontweight='bold')
    ax1.set_title('MAE per Test Engine', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, mae in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{mae:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # R² bar chart
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(esns)), r2s, color='#A23B72', alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(esns)))
    ax2.set_xticklabels([f'ESN {e}' for e in esns], fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('R² per Test Engine', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, r2 in zip(bars, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Scatter all folds
    ax3 = axes[1, 0]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, fr in enumerate(fold_results):
        ax3.scatter(fr['y_true'], fr['y_pred'], s=30, alpha=0.6, 
                   color=colors[i % len(colors)], label=f"ESN {fr['left_out_esn']}", 
                   edgecolors='black', linewidth=0.5)
    all_y = np.concatenate([fr['y_true'] for fr in fold_results])
    lims = [0, max(all_y) * 1.05]
    ax3.plot(lims, lims, 'r--', linewidth=2.5, alpha=0.8, label='Perfect')
    ax3.set_xlabel('Actual RUL', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted RUL', fontsize=12, fontweight='bold')
    ax3.set_title('Actual vs Predicted (All Folds)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Stats panel
    ax4 = axes[1, 1]
    ax4.axis('off')
    mae_mean = np.mean(maes)
    mae_std = np.std(maes, ddof=1) if len(maes) > 1 else 0
    r2_mean = np.mean(r2s)
    r2_std = np.std(r2s, ddof=1) if len(r2s) > 1 else 0
    
    stats = f"LOEO RECAP\n{'='*35}\n"
    stats += f"Window: {fold_results[0]['window']}\n"
    stats += f"Folds: {len(fold_results)}\n\n"
    stats += f"MAE: {mae_mean:.1f}±{mae_std:.1f}\n"
    stats += f"R²: {r2_mean:.3f}±{r2_std:.3f}\n\n"
    stats += "Per-fold:\n"
    for fr in fold_results:
        stats += f"ESN {fr['left_out_esn']}: {fr['mae']:.1f}, {fr['r2']:.3f}\n"
    
    ax4.text(0.1, 0.95, stats, transform=ax4.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.7))
    
    plt.suptitle(f"LOEO WW Summary | Window={fold_results[0]['window']}", 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fname = f"loeo_ww_summary_window{fold_results[0]['window']}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fname}")


def save_loeo_csv(fold_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    rows = [{
        'left_out_esn': fr['left_out_esn'], 'window': fr['window'], 
        'mae': fr['mae'], 'r2': fr['r2'], 'corr': fr['corr'],
        'train_time_s': fr['train_time_s'], 'inference_time_s': fr['inference_time_s'], 
        'n_test_seq': fr['n_test_seq']
    } for fr in fold_results]
    
    df_res = pd.DataFrame(rows).sort_values('left_out_esn')
    df_res.to_csv(os.path.join(out_dir, 'loeo_ww_results.csv'), index=False)
    
    recap = {
        'mae_mean': df_res['mae'].mean(),
        'mae_std': df_res['mae'].std(ddof=1) if len(df_res) > 1 else 0.0,
        'r2_mean': df_res['r2'].mean(),
        'r2_std': df_res['r2'].std(ddof=1) if len(df_res) > 1 else 0.0,
        'corr_mean': df_res['corr'].mean()
    }
    pd.DataFrame([recap]).to_csv(os.path.join(out_dir, 'loeo_ww_recap.csv'), index=False)
    
    print("\n✅ Saved: loeo_ww_results.csv, loeo_ww_recap.csv")
    print("\n" + "="*70)
    print("LOEO RESULTS")
    print("="*70)
    print(df_res.to_string(index=False))
    print("\n" + "="*70)
    print("RECAP (mean±std)")
    print("="*70)
    print(f"MAE: {recap['mae_mean']:.1f}±{recap['mae_std']:.1f} cycles")
    print(f"R²: {recap['r2_mean']:.3f}±{recap['r2_std']:.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("LSTM+Attention (WW) with LOEO")
    print("="*70)
    
    cfg = load_config('configs/config.yaml')
    df = pd.read_csv(cfg['data']['train_clean_csv'])
    print(f"\nLoaded: {df.shape}, ESNs: {df['ESN'].nunique()}")
    
    df_eng = engineer_ww_features(df)
    diagnose_data(df_eng)  

    
    feature_cols = [
        'temp_diff_norm', 'thermal_stress', 'load_factor', 'speed_ratio', 
        'temp_gradient', 'pressure_ratio', 'hptc_efficiency', 'hpt_stress_indicator',
        'Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_Pt2', 'Sensed_TAT', 
        'Sensed_WFuel', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 
        'Sensed_Core_Speed', 'Sensed_T25', 'Sensed_T3', 'Sensed_Ps3', 'Sensed_T45'
    ]
    
    print(f"\n{'='*70}\nFEATURES: {len(feature_cols)} (8 engineered + 14 raw)\n{'='*70}")
    
    window_size = 100
    fold_results = train_ww_loeo(df_eng, feature_cols, window_size, 
                                 log_target=True, epochs=300, batch_size=32)
    
    if not fold_results:
        print("\n❌ No valid folds")
        return
    
    out_dir = f'artifacts/loeo_ww_lstm_window{window_size}'
    print(f"\n{'='*70}\nSAVING RESULTS\n{'='*70}")
    
    plot_loeo_predictions(fold_results, out_dir)
    plot_loeo_summary(fold_results, out_dir)
    save_loeo_csv(fold_results, out_dir)
    
    print(f"\n📁 All results: {out_dir}/")
    print("\n" + "="*70)
    print("DONE ✨")
    print("="*70)


if __name__ == '__main__':
    main()
