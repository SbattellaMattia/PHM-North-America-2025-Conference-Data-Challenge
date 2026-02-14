"""
TCN + Attention per predizione MULTI-TASK (WW, HPC_SV, HPT_SV)
================================================================

STRUTTURA:
1. Feature Engineering: identica a LSTM version
2. Sequence Creation: identica a LSTM version
3. Model Building: TCN + Attention + 3 output heads
4. Training: multi-task learning
5. Visualization: plot risultati + confronto con LSTM

DIFFERENZE RISPETTO A LSTM:
- TCN (Temporal Convolutional Network) invece di LSTM
- Convoluzioni causali dilate invece di recurrence
- Training più veloce (parallelizzazione GPU)
- Gradiente più stabile

AUTORE: TCN Multi-Task Learning Version
DATA: 2026-02-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import os
from src.utils import load_config
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================================
# SEZIONE 1: FEATURE ENGINEERING (identica a LSTM)
# ============================================================================

def create_base_features(df):
    """Crea feature ingegnerizzate da sensori grezzi."""
    print("  Creating engineered features...")

    df['temp_diff_norm'] = (
        (df['Sensed_T45'] - df['Sensed_T25']) /
        (df['Sensed_T25'] + 1e-6)
    ) ** 2

    df['thermal_stress'] = (
        (df['Sensed_T45'] - df['Sensed_T25']) *
        df['Sensed_Core_Speed'] / 1000
    )

    df['load_factor'] = (
        df['Sensed_WFuel'] * df['Sensed_Core_Speed']
    )

    df['speed_ratio'] = (
        df['Sensed_Core_Speed'] /
        (df['Sensed_Fan_Speed'] + 1e-6)
    )

    df['temp_gradient'] = (
        (df['Sensed_T45'] - df['Sensed_T3']) /
        (df['Sensed_T3'] + 1e-6)
    )

    df['pressure_ratio'] = (
        df['Sensed_Ps3'] /
        (df['Sensed_TAT'] + 1e-6)
    )

    print(f"  Created 6 engineered features")
    return df

def aggregate_by_cycle(df):
    """Aggrega 8 snapshot per ciclo → 1 valore per ciclo."""
    print("  Aggregating 8 snapshots per cycle...")

    exclude_cols = ['ESN', 'Cycles', 'Snapshot',
                    'Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    agg_dict = {col: 'mean' for col in feature_cols}
    agg_dict['Cycles_to_WW'] = 'first'
    agg_dict['Cycles_to_HPC_SV'] = 'first'
    agg_dict['Cycles_to_HPT_SV'] = 'first'

    df_agg = df.groupby(['ESN', 'Cycles']).agg(agg_dict).reset_index()
    df_agg = df_agg.sort_values(['ESN', 'Cycles']).reset_index(drop=True)

    print(f"  Before: {len(df)} rows → After: {len(df_agg)} rows")
    return df_agg

def engineer_ww_features(df):
    """Pipeline completa feature engineering."""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)

    print(f"\nInput shape: {df.shape}")
    print(f"  ESNs: {df['ESN'].nunique()}")

    df = create_base_features(df)
    df_agg = aggregate_by_cycle(df)

    print("\n  Cleaning invalid values...")
    df_clean = df_agg.replace([np.inf, -np.inf], np.nan)
    rows_before = len(df_clean)
    df_clean = df_clean.dropna()
    rows_dropped = rows_before - len(df_clean)

    if rows_dropped > 0:
        print(f"  Dropped {rows_dropped} rows with NaN/inf")
    else:
        print(f"  ✅ No invalid values found")

    print(f"\nFinal shape: {df_clean.shape}")
    return df_clean

# ============================================================================
# SEZIONE 2: SEQUENCE CREATION (identica a LSTM)
# ============================================================================

def create_sequences_multitask(df, feature_cols, window_size):
    """Crea sequenze temporali con 3 target (WW, HPC_SV, HPT_SV)."""
    print(f"\n  Creating multi-task sequences (window={window_size})...")

    X_seq = []
    y_ww, y_hpc, y_hpt = [], [], []
    cycle_info = []

    for esn in sorted(df['ESN'].unique()):
        d_esn = df[df['ESN'] == esn].sort_values('Cycles').reset_index(drop=True)

        X = d_esn[feature_cols].values
        y_ww_esn = d_esn['Cycles_to_WW'].values
        y_hpc_esn = d_esn['Cycles_to_HPC_SV'].values
        y_hpt_esn = d_esn['Cycles_to_HPT_SV'].values
        cycles = d_esn['Cycles'].values

        n_sequences = len(X) - window_size + 1

        for i in range(n_sequences):
            X_seq.append(X[i:i+window_size])
            y_ww.append(y_ww_esn[i+window_size-1])
            y_hpc.append(y_hpc_esn[i+window_size-1])
            y_hpt.append(y_hpt_esn[i+window_size-1])

            cycle_info.append({
                'ESN': esn,
                'Cycle_end': cycles[i+window_size-1],
                'Sequence_idx': len(X_seq) - 1
            })

        print(f"    ESN {esn}: {len(X)} cycles → {n_sequences} sequences")

    X_seq = np.array(X_seq)
    y_ww = np.array(y_ww)
    y_hpc = np.array(y_hpc)
    y_hpt = np.array(y_hpt)

    print(f"\n  Total sequences: {X_seq.shape[0]:,}")
    print(f"  Input shape: {X_seq.shape}")
    print(f"  WW range: [{y_ww.min():.0f}, {y_ww.max():.0f}]")
    print(f"  HPC range: [{y_hpc.min():.0f}, {y_hpc.max():.0f}]")
    print(f"  HPT range: [{y_hpt.min():.0f}, {y_hpt.max():.0f}]")

    return X_seq, {'ww': y_ww, 'hpc': y_hpc, 'hpt': y_hpt}, pd.DataFrame(cycle_info)

# ============================================================================
# SEZIONE 3: TCN ARCHITECTURE (NUOVA - sostituisce LSTM)
# ============================================================================

class AttentionLayer(layers.Layer):
    """Attention mechanism custom per Keras (identica a LSTM version)."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        output = tf.reduce_sum(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        return config

class ResidualBlock(layers.Layer):
    """
    Residual Block per TCN.

    STRUTTURA:
    ----------
    Input → Conv1D(dilation) → ReLU → Dropout → Conv1D(dilation) → ReLU → Dropout
              ↓                                                                ↓
              ╰─────────────────────── Skip Connection ──────────────────────╯
                                              ↓
                                           Output (Add)

    CAUSAL CONVOLUTION:
    -------------------
    Padding='causal' garantisce che l'output al tempo t dipenda solo da input ≤ t
    (nessuna "fuga di informazione" dal futuro)

    DILATION:
    ---------
    Espande il receptive field esponenzialmente:
    - dilation=1: vede 3 timesteps
    - dilation=2: vede 5 timesteps
    - dilation=4: vede 9 timesteps
    - dilation=8: vede 17 timesteps
    → Con 4 blocchi (dilation 1,2,4,8): receptive field = 31 timesteps
    """

    def __init__(self, filters, kernel_size, dilation_rate, dropout, l2_reg, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.l2_reg = l2_reg

        # Conv layers con causal padding
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )
        self.dropout2 = layers.Dropout(dropout)

        # Skip connection (1x1 conv se necessario)
        self.skip_conv = None

    def build(self, input_shape):
        # Se dimensioni diverse, usa 1x1 conv per matching
        if input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding='same',
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)
            )
        super(ResidualBlock, self).build(input_shape)

    def call(self, x, training=False):
        # Main path
        out = self.conv1(x)
        out = self.dropout1(out, training=training)
        out = self.conv2(out)
        out = self.dropout2(out, training=training)

        # Skip connection
        if self.skip_conv is not None:
            x = self.skip_conv(x)

        # Residual add
        return layers.add([x, out])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout': self.dropout,
            'l2_reg': self.l2_reg
        })
        return config

def build_tcn_attention_multitask(timesteps, n_features,
                                   tcn_filters=[64, 64, 32, 32],
                                   kernel_size=3,
                                   dense_units=[16],
                                   dropout=0.3,
                                   l2_reg=0.001):
    """
    TCN + Attention per predizione MULTI-TASK (3 RUL).

    ARCHITETTURA:
    -------------
    Input → TCN Block (d=1) → TCN Block (d=2) → TCN Block (d=4) → TCN Block (d=8)
                ↓                ↓                ↓                  ↓
            Residual         Residual         Residual          Residual
                                                                     ↓
                                                              Attention
                                                                     ↓
                                                              Shared Dense
                                                                     ↓
                            ┌────────────────────────────────────────┼─────────┐
                            ↓                                        ↓         ↓
                        WW Head                                  HPC Head  HPT Head
                            ↓                                        ↓         ↓
                        Output WW                               Output HPC Output HPT

    RECEPTIVE FIELD:
    ----------------
    Con kernel_size=3 e dilation=[1,2,4,8]:
    RF = 1 + 2*(kernel_size-1)*sum(dilations)
       = 1 + 2*(3-1)*(1+2+4+8)
       = 1 + 4*15 = 61 timesteps

    → Con window=30, TCN può vedere TUTTA la sequenza + contesto extra!

    VANTAGGI VS LSTM:
    -----------------
    1. Parallelizzazione completa (GPU fully utilized)
    2. Gradiente stabile (no vanishing gradient)
    3. Receptive field controllabile via dilation
    4. Training 2-3× più veloce

    SVANTAGGI VS LSTM:
    ------------------
    1. Più parametri (~2× con stessa capacità)
    2. Meno "memoria" intrinseca (no hidden state)
    3. Può richiedere più tuning (kernel, dilation)
    """
    print(f"\n  Building Multi-Task TCN+Attention model...")
    print(f"    Input: ({timesteps} timesteps, {n_features} features)")
    print(f"    TCN filters: {tcn_filters}")
    print(f"    Kernel size: {kernel_size}")
    print(f"    Dense layers: {dense_units}")

    # Input layer
    inputs = keras.Input(shape=(timesteps, n_features), name='input')

    # ========================================================================
    # SHARED TCN LAYERS (sostituiscono LSTM)
    # ========================================================================

    x = inputs

    # Stack di residual blocks con dilation crescente
    dilations = [1, 2, 4, 8]  # Esponenziale: 1, 2, 4, 8

    for i, (filters, dilation) in enumerate(zip(tcn_filters, dilations)):
        x = ResidualBlock(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout=dropout,
            l2_reg=l2_reg,
            name=f'tcn_block_{i+1}'
        )(x)
        print(f"    TCN block {i+1}: {filters} filters, dilation={dilation}")

    # Calcola receptive field
    receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    print(f"\n    Receptive field: {receptive_field} timesteps")
    if receptive_field >= timesteps:
        print(f"    ✅ RF covers entire window ({timesteps} timesteps)")
    else:
        print(f"    ⚠️ RF smaller than window (coverage: {receptive_field/timesteps*100:.1f}%)")

    # Attention (collassa timesteps) - IDENTICA A LSTM VERSION
    x = AttentionLayer(name='shared_attention')(x)
    print(f"    Attention layer: collapse {timesteps} timesteps → 1")

    # Shared dense layer
    shared = layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='shared_dense'
    )(x)
    shared = layers.Dropout(dropout, name='dropout_shared')(shared)
    print(f"    Shared dense: 32 units")

    # ========================================================================
    # TASK-SPECIFIC HEADS (identici a LSTM version)
    # ========================================================================

    print(f"\n    Task-specific heads:")

    # HEAD 1: Wear and Tear (WW)
    ww_dense = layers.Dense(
        16,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='ww_dense'
    )(shared)
    ww_dense = layers.Dropout(dropout, name='ww_dropout')(ww_dense)
    output_ww = layers.Dense(1, activation='linear', name='output_ww')(ww_dense)
    print(f"      WW head: 16 → 1")

    # HEAD 2: High Pressure Compressor (HPC)
    hpc_dense = layers.Dense(
        16,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='hpc_dense'
    )(shared)
    hpc_dense = layers.Dropout(dropout, name='hpc_dropout')(hpc_dense)
    output_hpc = layers.Dense(1, activation='linear', name='output_hpc')(hpc_dense)
    print(f"      HPC head: 16 → 1")

    # HEAD 3: High Pressure Turbine (HPT)
    hpt_dense = layers.Dense(
        16,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='hpt_dense'
    )(shared)
    hpt_dense = layers.Dropout(dropout, name='hpt_dropout')(hpt_dense)
    output_hpt = layers.Dense(1, activation='linear', name='output_hpt')(hpt_dense)
    print(f"      HPT head: 16 → 1")

    # Model con 3 output
    model = Model(
        inputs=inputs,
        outputs=[output_ww, output_hpc, output_hpt],
        name='TCN_Attention_MultiTask'
    )

    # Compile (identico a LSTM)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'output_ww': 'mse',
            'output_hpc': 'mse',
            'output_hpt': 'mse'
        },
        loss_weights={
            'output_ww': 1.0,
            'output_hpc': 0.5,
            'output_hpt': 0.7
        },
        metrics={
            'output_ww': ['mae'],
            'output_hpc': ['mae'],
            'output_hpt': ['mae']
        }
    )

    total_params = model.count_params()
    print(f"\n    Total parameters: {total_params:,}")

    return model

# ============================================================================
# SEZIONE 4: TRAINING (con timing per confronto)
# ============================================================================

def train_multitask_model(df, feature_cols, window_size):
    """Training completo multi-task con TCN."""
    print(f"\n{'='*70}")
    print(f"TRAINING MULTI-TASK TCN: Window Size = {window_size} cycles")
    print('='*70)

    # Step 1: Crea sequenze
    X_seq, y_dict, info = create_sequences_multitask(df, feature_cols, window_size)

    if X_seq.shape[0] < 100:
        print(f"\n⚠️ Too few sequences, skipping")
        return None

    # Step 2: Log-transform dei 3 target
    print("\n  Transforming targets to log-scale...")
    y_ww_log = np.log(y_dict['ww'] + 1)
    y_hpc_log = np.log(y_dict['hpc'] + 1)
    y_hpt_log = np.log(y_dict['hpt'] + 1)

    # Step 3: Standardizza features
    print("\n  Standardizing features...")
    scaler = StandardScaler()
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(X_seq.shape)

    # Step 4: Split train/val
    print("\n  Splitting train/validation (80/20)...")
    indices = np.arange(len(X_scaled))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_ww_train, y_ww_val = y_ww_log[train_idx], y_ww_log[val_idx]
    y_hpc_train, y_hpc_val = y_hpc_log[train_idx], y_hpc_log[val_idx]
    y_hpt_train, y_hpt_val = y_hpt_log[train_idx], y_hpt_log[val_idx]

    print(f"    Train: {len(train_idx):,} sequences")
    print(f"    Val: {len(val_idx):,} sequences")

    # Step 5: Build TCN model
    model = build_tcn_attention_multitask(
        timesteps=window_size,
        n_features=X_seq.shape[-1]
    )

    # Step 6: Train (con timing)
    print("\n  Training TCN model...")
    print("    Max epochs: 300")
    print("    Batch size: 32")

    start_time = time.time()

    history = model.fit(
        X_train,
        {
            'output_ww': y_ww_train,
            'output_hpc': y_hpc_train,
            'output_hpt': y_hpt_train
        },
        validation_data=(
            X_val,
            {
                'output_ww': y_ww_val,
                'output_hpc': y_hpc_val,
                'output_hpt': y_hpt_val
            }
        ),
        epochs=300,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ],
        verbose=2
    )

    training_time = time.time() - start_time

    print(f"\n  Training completed: {len(history.history['loss'])} epochs")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")

    # Step 7: Evaluate (con timing inference)
    print("\n  Evaluating on all data...")

    start_inference = time.time()
    y_pred_ww_log, y_pred_hpc_log, y_pred_hpt_log = model.predict(X_scaled, verbose=0)
    inference_time = time.time() - start_inference

    print(f"  Inference time: {inference_time:.3f}s ({inference_time/len(X_scaled)*1000:.3f}ms per sequence)")

    # Inverse transform
    y_pred_ww = np.clip(np.exp(y_pred_ww_log.flatten()) - 1, 0, None)
    y_pred_hpc = np.clip(np.exp(y_pred_hpc_log.flatten()) - 1, 0, None)
    y_pred_hpt = np.clip(np.exp(y_pred_hpt_log.flatten()) - 1, 0, None)

    # Metrics per ogni target
    results = {
        'WW': {
            'mae': mean_absolute_error(y_dict['ww'], y_pred_ww),
            'r2': r2_score(y_dict['ww'], y_pred_ww),
            'y_true': y_dict['ww'],
            'y_pred': y_pred_ww
        },
        'HPC_SV': {
            'mae': mean_absolute_error(y_dict['hpc'], y_pred_hpc),
            'r2': r2_score(y_dict['hpc'], y_pred_hpc),
            'y_true': y_dict['hpc'],
            'y_pred': y_pred_hpc
        },
        'HPT_SV': {
            'mae': mean_absolute_error(y_dict['hpt'], y_pred_hpt),
            'r2': r2_score(y_dict['hpt'], y_pred_hpt),
            'y_true': y_dict['hpt'],
            'y_pred': y_pred_hpt
        }
    }

    print("\n  ✅ RESULTS (Multi-Task TCN):")
    for target, metrics in results.items():
        corr = np.sqrt(metrics['r2']) if metrics['r2'] > 0 else 0
        print(f"    {target:8s}: MAE={metrics['mae']:7.1f} cycles, R²={metrics['r2']:.3f}, Corr={corr:.3f}")

    return {
        'window': window_size,
        'results': results,
        'model': model,
        'scaler': scaler,
        'history': history,
        'X': X_scaled,
        'info': info,
        'training_time': training_time,
        'inference_time': inference_time
    }

# ============================================================================
# SEZIONE 5: VISUALIZATION (identica + timing comparison)
# ============================================================================

def plot_multitask_comparison(result, out_dir, model_type='TCN'):
    """Plot confronto 3 target."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    window = result['window']
    info = result['info']

    targets = ['WW', 'HPC_SV', 'HPT_SV']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    esn_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for row, (target, color) in enumerate(zip(targets, colors)):
        metrics = result['results'][target]
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        mae = metrics['mae']
        r2 = metrics['r2']
        corr = np.sqrt(r2) if r2 > 0 else 0

        # Column 1: Scatter plot
        ax_scatter = axes[row, 0]

        scatter = ax_scatter.scatter(
            y_true, y_pred,
            c=np.arange(len(y_true)),
            cmap='viridis',
            s=20,
            alpha=0.6,
            edgecolors='none'
        )

        lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
        ax_scatter.plot(lims, lims, 'r--', linewidth=2.5, alpha=0.8, label='Perfect prediction')

        textstr = f'MAE: {mae:.1f} cycles\nR²: {r2:.3f}\nCorr: {corr:.3f}\nN: {len(y_true):,}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.95)
        ax_scatter.text(0.05, 0.95, textstr, transform=ax_scatter.transAxes,
                       fontsize=11, verticalalignment='top', bbox=props, fontweight='bold')

        ax_scatter.set_xlabel('Actual RUL (cycles)', fontsize=12, fontweight='bold')
        ax_scatter.set_ylabel('Predicted RUL (cycles)', fontsize=12, fontweight='bold')
        ax_scatter.set_title(f'{target} - Actual vs Predicted', fontsize=13, fontweight='bold', color=color)
        ax_scatter.legend(fontsize=10, loc='lower right')
        ax_scatter.grid(True, alpha=0.3, linestyle='--')
        ax_scatter.set_xlim(lims)
        ax_scatter.set_ylim(lims)

        # Column 2: Time series
        ax_time = axes[row, 1]

        for i, esn in enumerate(sorted(info['ESN'].unique())): 
            mask = info['ESN'] == esn
            cycles = info[mask]['Cycle_end'].values
            y_true_esn = y_true[mask]
            y_pred_esn = y_pred[mask]

            esn_color = esn_colors[i % len(esn_colors)]  

            ax_time.plot(cycles, y_true_esn, '-o',  
            label=f'ESN {esn} - Actual',
            color=esn_color,  
            markersize=3,
            linewidth=2,
            alpha=0.8)
            ax_time.plot(cycles, y_pred_esn, '--',  
            label=f'ESN {esn} - Predicted',
            color=esn_color,
            linewidth=2.5,  
            alpha=0.7)

        ax_time.set_xlabel('Cycle', fontsize=12, fontweight='bold')
        ax_time.set_ylabel('RUL (cycles)', fontsize=12, fontweight='bold')
        ax_time.set_title(f'{target} - Time Series Evolution', fontsize=13, fontweight='bold', color=color)
        ax_time.legend(fontsize=9, loc='best', ncol=2)
        ax_time.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(
        f'Multi-Task {model_type}+Attention | Window={window} cycles | All 3 Components',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fname = f"multitask_{model_type.lower()}_predictions_window{window}.png"
    fig.savefig(f"{out_dir}/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {fname}")

def plot_multitask_summary(result, out_dir, model_type='LSTM'):
    """
    Plot summary DETTAGLIATO con layout ottimizzato:
    - Prima riga: MAE, R², Stats
    - Seconda riga: ESN 101, 102, 103, 104 (riquadri separati)
    """
    fig = plt.figure(figsize=(28, 10))  # Wide per 4 ESN

    window = result['window']
    targets = ['WW', 'HPC_SV', 'HPT_SV']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    esn_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # ========================================================================
    # PRIMA RIGA: Metriche aggregate
    # ========================================================================

    # MAE Bar Chart
    ax1 = plt.subplot(2, 4, 1)
    maes = [result['results'][t]['mae'] for t in targets]
    bars = ax1.bar(targets, maes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{mae:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MAE (cycles)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # R² Bar Chart
    ax2 = plt.subplot(2, 4, 2)
    r2s = [result['results'][t]['r2'] for t in targets]
    bars = ax2.bar(targets, r2s, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, r2 in zip(bars, r2s):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{r2:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('R² Score', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, max(r2s) * 1.15])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Per-ESN MAE Comparison
    ax3 = plt.subplot(2, 4, 3)
    info = result['info']
    all_esns = sorted(info['ESN'].unique())
    esn_avg_maes = []

    for esn in all_esns:
        mask = info['ESN'] == esn
        esn_maes = []
        for target in targets:
            y_true = result['results'][target]['y_true'][mask]
            y_pred = result['results'][target]['y_pred'][mask]
            mae_esn = np.mean(np.abs(y_true - y_pred))
            esn_maes.append(mae_esn)
        esn_avg_maes.append(np.mean(esn_maes))

    bars = ax3.bar([f'ESN {esn}' for esn in all_esns], esn_avg_maes, 
                   color=esn_colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, mae in zip(bars, esn_avg_maes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{mae:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average MAE (cycles)', fontsize=12, fontweight='bold')
    ax3.set_title('MAE per Engine', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Stats Panel
    ax4 = plt.subplot(2, 4, 4)
    ax4.axis('off')
    train_time = result.get('training_time', 0)
    inf_time = result.get('inference_time', 0)
    n_sequences = len(result['X'])

    stats_text = f"""
MODEL: {model_type}
Window: {window} cycles
N Sequences: {n_sequences:,}

TIMING:
Training: {train_time:.1f}s
Inference: {inf_time:.3f}s
Speed: {inf_time/n_sequences*1000:.2f} ms/seq

METRICS (AVG):
MAE: {np.mean(maes):.1f} cycles
R²: {np.mean(r2s):.3f}
Correlation: {np.mean([np.sqrt(r2) for r2 in r2s]):.3f}
    """
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.5))

    # ========================================================================
    # SECONDA RIGA: Time series per ogni ESN (4 riquadri)
    # ========================================================================
    markers = ['o', 's', '^']

    for idx, esn in enumerate(all_esns):
        ax = plt.subplot(2, 4, 5 + idx)

        mask = info['ESN'] == esn
        cycles = info[mask]['Cycle_end'].values
        esn_color = esn_colors[idx]

        for j, target in enumerate(targets):
            y_true = result['results'][target]['y_true'][mask]
            y_pred = result['results'][target]['y_pred'][mask]

            # Actual
            ax.plot(cycles, y_true, f'-{markers[j]}',
                   label=f'{target}',
                   color=colors[j],
                   markersize=3.5,
                   linewidth=1.8,
                   alpha=0.85,
                   markeredgecolor=esn_color,
                   markeredgewidth=1.2)

            # Predicted
            ax.plot(cycles, y_pred, '--',
                   color=colors[j],
                   linewidth=2.2,
                   alpha=0.55)

        # MAE per questo ESN
        esn_mae = esn_avg_maes[idx]

        ax.set_xlabel('Cycle', fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('RUL (cycles)', fontsize=10, fontweight='bold')
        ax.set_title(f'ESN {esn} (MAE: {esn_mae:.1f})', 
                    fontsize=12, fontweight='bold', color=esn_color)

        # Legenda solo nel primo plot
        if idx == 0:
            ax.legend(fontsize=8, loc='best', framealpha=0.9,
                     title='Solid=Actual, Dash=Pred')

        ax.grid(True, alpha=0.3, linestyle='--')

        # Bordo colorato
        for spine in ax.spines.values():
            spine.set_edgecolor(esn_color)
            spine.set_linewidth(2)

    if train_time > 0:
        title = f'Multi-Task {model_type} | Window={window} | Train: {train_time:.1f}s | Inference: {inf_time:.3f}s | Total MAE: {np.mean(esn_avg_maes):.1f}'
    else:
        title = f'Multi-Task {model_type} Summary | Window={window} cycles'

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"multitask_{model_type.lower()}_summary_window{window}.png"
    fig.savefig(f"{out_dir}/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fname}")

# ============================================================================
# SEZIONE 6: MAIN
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print("TCN + ATTENTION FOR MULTI-TASK RUL PREDICTION")
    print("Predicting WW, HPC_SV, HPT_SV simultaneously")
    print("="*70)

    # Load data
    print("\nLoading data...")
    cfg = load_config("configs/config.yaml")
    df = pd.read_csv(cfg["data"]["train_clean_csv"])
    print(f"  Raw data shape: {df.shape}")
    print(f"  ESNs: {df['ESN'].nunique()}")

    # Feature engineering
    df_eng = engineer_ww_features(df)

    # Define feature columns (le stesse del tuo codice)
    feature_cols = [
        'temp_diff_norm',
        'thermal_stress',
        'load_factor',
        'speed_ratio',
        'temp_gradient',
        'pressure_ratio',
        'ratio_T3_T45',
        'tri_ratio_diff_Sensed_Mach_Sensed_T3_Sensed_T45',
        'HPC_Eff_Index_clean',
        'Sensed_Altitude',
        'Sensed_Mach',
        'Sensed_Pamb',
        'Sensed_Pt2',
        'Sensed_TAT',
        'Sensed_WFuel',
        'Sensed_VAFN',
        'Sensed_VBV',
        'Sensed_Fan_Speed',
        'Sensed_Core_Speed',
        'Sensed_T25',
        'Sensed_T3',
        'Sensed_Ps3',
        'Sensed_T45',
    ]

    print(f"\n{'='*70}")
    print(f"FEATURES: {len(feature_cols)} total")
    print(f"{'='*70}")

    # Train TCN model
    window_size = 200
    print(f"\n{'='*70}")
    print(f"TRAINING WINDOW SIZE: {window_size}")
    print(f"{'='*70}")

    result = train_multitask_model(df_eng, feature_cols, window_size)

    if result is None:
        print("\n❌ Training failed")
        return

    # Summary
    print("\n" + "="*70)
    print("🏆 FINAL RESULTS - MULTI-TASK TCN")
    print("="*70)
    print(f"\nWindow Size: {result['window']} cycles")
    print(f"Training Time: {result['training_time']:.1f}s ({result['training_time']/60:.1f} min)")
    print(f"Inference Time: {result['inference_time']:.3f}s ({result['inference_time']/len(result['X'])*1000:.3f}ms/seq)")
    print(f"\nPer-Component Performance:")
    for target in ['WW', 'HPC_SV', 'HPT_SV']:
        metrics = result['results'][target]
        corr = np.sqrt(metrics['r2']) if metrics['r2'] > 0 else 0
        print(f"\n  {target}:")
        print(f"    MAE: {metrics['mae']:.2f} cycles")
        print(f"    R²: {metrics['r2']:.4f}")
        print(f"    Correlation: {corr:.4f}")

    # Save results
    out_dir = f"artifacts/multitask_tcn_window{window_size}"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Plots
    plot_multitask_comparison(result, out_dir, model_type='TCN')
    plot_multitask_summary(result, out_dir, model_type='TCN')

    # Model
    model_path = f"{out_dir}/multitask_tcn_model_window{result['window']}.h5"
    result['model'].save(model_path)
    print(f"\n✅ Model saved: multitask_tcn_model_window{result['window']}.h5")

    # Scaler
    import joblib
    scaler_path = f"{out_dir}/scaler_window{result['window']}.pkl"
    joblib.dump(result['scaler'], scaler_path)
    print(f"✅ Scaler saved: scaler_window{result['window']}.pkl")

    # CSV results (con timing)
    results_df = pd.DataFrame([
        {
            'Component': target,
            'MAE': result['results'][target]['mae'],
            'R2': result['results'][target]['r2'],
            'Correlation': np.sqrt(result['results'][target]['r2']) if result['results'][target]['r2'] > 0 else 0
        }
        for target in ['WW', 'HPC_SV', 'HPT_SV']
    ])
    results_df.to_csv(f"{out_dir}/multitask_tcn_results.csv", index=False)
    print(f"✅ Results saved: multitask_tcn_results.csv")

    # Save timing stats
    timing_df = pd.DataFrame([{
        'Model': 'TCN',
        'Window': result['window'],
        'Training_Time_s': result['training_time'],
        'Inference_Time_s': result['inference_time'],
        'Inference_ms_per_seq': result['inference_time'] / len(result['X']) * 1000
    }])
    timing_df.to_csv(f"{out_dir}/tcn_timing_stats.csv", index=False)
    print(f"✅ Timing stats saved: tcn_timing_stats.csv")

    print(f"\n📁 All results saved in: {out_dir}/")

    print("\n" + "="*70)
    print("DONE ✨ - Multi-Task TCN Prediction Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
