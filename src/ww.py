"""
LSTM + Attention per predizione WW (Wear and Tear)
==================================================

STRUTTURA:
1. Feature Engineering: crea feature da sensori grezzi (TUTTI gli 8 snapshot)
2. Sequence Creation: trasforma in sequenze temporali
3. Model Building: LSTM + Attention
4. Training: test window sizes variabili
5. Visualization: plot risultati

DATI:
- Ogni ciclo ha 8 snapshot (8 istanti diversi durante il volo)
- Ogni snapshot ha ~15 sensori
- Aggreghiamo 8 snapshot → 1 valore per ciclo (media)

AUTORE: [tuo nome]
DATA: 2026-02-12
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
warnings.filterwarnings('ignore')


# ============================================================================
# SEZIONE 1: FEATURE ENGINEERING
# ============================================================================

def create_base_features(df):
    """
    Crea feature ingegnerizzate da sensori grezzi.
    
    COSA FA:
    --------
    Prende sensori grezzi (temperatura, pressione, velocità) e crea
    combinazioni fisicamente significative che catturano:
    - Stress termico (temperatura × velocità)
    - Carico meccanico (fuel × velocità)
    - Inefficienze (rapporti di velocità/pressione/temperatura)
    
    INPUT:
    ------
    df: DataFrame con colonne:
        - Sensed_* (sensori grezzi)
        - ESN (motore ID)
        - Cycles (ciclo operativo)
        - Snapshot (1-8, istante durante il volo)
        - Cycles_to_WW (target - remaining useful life)
    
    OUTPUT:
    -------
    df: DataFrame con 6 feature aggiuntive (colonne nuove)
    
    FEATURE CREATE:
    ---------------
    1. temp_diff_norm: stress termico normalizzato
       Formula: [(T45 - T25) / T25]²
       Unità: adimensionale
       Range tipico: 0.1 - 1.0
       Fisica: T45 (post-combustore) vs T25 (pre-compressore)
               Grande differenza → alta combustione → stress termico
    
    2. thermal_stress: stress termico assoluto
       Formula: (T45 - T25) × Core_Speed / 1000
       Unità: K·RPM/1000
       Range tipico: 50 - 500
       Fisica: Alta temperatura + alta velocità → usura accelerata
    
    3. load_factor: carico motore totale
       Formula: WFuel × Core_Speed
       Unità: kg/s × RPM
       Range tipico: 10000 - 100000
       Fisica: Più fuel + più velocità → più potenza → più stress
    
    4. speed_ratio: rapporto velocità core/fan
       Formula: Core_Speed / Fan_Speed
       Unità: adimensionale
       Range tipico: 1.5 - 3.0
       Fisica: Disallineamento tra core e fan → vibrazione → usura
    
    5. temp_gradient: gradiente temperatura combustione
       Formula: (T45 - T3) / T3
       Unità: adimensionale
       Range tipico: 0.5 - 2.0
       Fisica: Cambio temperatura troppo rapido → stress termico
    
    6. pressure_ratio: rapporto compressione
       Formula: Ps3 / TAT
       Unità: Pa/K (approssimato)
       Range tipico: 10 - 100
       Fisica: Alta compressione → alta efficienza ma anche stress
    """
    print("  Creating engineered features...")
    
    # Feature 1: Stress termico normalizzato
    df['temp_diff_norm'] = (
        (df['Sensed_T45'] - df['Sensed_T25']) / 
        (df['Sensed_T25'] + 1e-6)  # +1e-6 evita divisione per zero
    ) ** 2
    
    # Feature 2: Stress termico assoluto
    df['thermal_stress'] = (
        (df['Sensed_T45'] - df['Sensed_T25']) * 
        df['Sensed_Core_Speed'] / 1000  # /1000 per scaling
    )
    
    # Feature 3: Carico motore
    df['load_factor'] = (
        df['Sensed_WFuel'] * df['Sensed_Core_Speed']
    )
    
    # Feature 4: Rapporto velocità
    df['speed_ratio'] = (
        df['Sensed_Core_Speed'] / 
        (df['Sensed_Fan_Speed'] + 1e-6)
    )
    
    # Feature 5: Gradiente temperatura
    df['temp_gradient'] = (
        (df['Sensed_T45'] - df['Sensed_T3']) / 
        (df['Sensed_T3'] + 1e-6)
    )
    
    # Feature 6: Rapporto pressioni
    df['pressure_ratio'] = (
        df['Sensed_Ps3'] / 
        (df['Sensed_TAT'] + 1e-6)
    )
    
    print(f"    Created 6 engineered features")
    
    return df


def aggregate_by_cycle(df):
    """
    Aggrega 8 snapshot per ciclo → 1 valore per ciclo.
    
    COSA FA:
    --------
    Ogni ciclo operativo ha 8 snapshot corrispondenti a 8 momenti
    diversi durante il volo (es. decollo, crociera, atterraggio, ecc.).
    
    Questa funzione calcola la MEDIA di ogni sensore/feature attraverso
    gli 8 snapshot, producendo 1 singolo valore "rappresentativo" del ciclo.
    
    PERCHÉ LA MEDIA:
    ----------------
    1. Robustezza al rumore: outlier in 1 snapshot hanno meno impatto
    2. Rappresentatività: cattura comportamento "tipico" del ciclo
    3. Dimensionalità: riduce dati da N×8 a N (più gestibile per LSTM)
    
    ESEMPIO:
    --------
    Prima (8 snapshot):
      ESN=101, Cycle=1000, Snapshot=1: T45=800K, Core_Speed=9000
      ESN=101, Cycle=1000, Snapshot=2: T45=820K, Core_Speed=9100
      ESN=101, Cycle=1000, Snapshot=3: T45=810K, Core_Speed=9050
      ...
      ESN=101, Cycle=1000, Snapshot=8: T45=805K, Core_Speed=9020
    
    Dopo (1 valore):
      ESN=101, Cycle=1000: T45=809K (media), Core_Speed=9040 (media)
    
    INPUT:
    ------
    df: DataFrame con 8 righe per ciclo (1 per snapshot)
        Shape: (N_cycles × 8, N_columns)
    
    OUTPUT:
    -------
    df_agg: DataFrame con 1 riga per ciclo
            Shape: (N_cycles, N_columns)
    
    NOTE:
    -----
    - Il target (Cycles_to_WW) è uguale per tutti gli 8 snapshot
      dello stesso ciclo, quindi usiamo 'first' invece di 'mean'
    """
    print("  Aggregating 8 snapshots per cycle...")
    
    # Identifica colonne da aggregare (escludi metadata e target)
    exclude_cols = ['ESN', 'Cycles', 'Snapshot', 
                   'Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"    Features to aggregate: {len(feature_cols)}")
    
    # Dizionario aggregazione
    # - feature: calcola MEDIA attraverso 8 snapshot
    # - target: prendi PRIMO valore (tutti uguali)
    agg_dict = {col: 'mean' for col in feature_cols}
    agg_dict['Cycles_to_WW'] = 'first'
    
    # Aggrega per (ESN, Cycles)
    df_agg = df.groupby(['ESN', 'Cycles']).agg(agg_dict).reset_index()
    
    # Ordina per ESN e Cycles (CRITICO per sequenze temporali)
    df_agg = df_agg.sort_values(['ESN', 'Cycles']).reset_index(drop=True)
    
    print(f"    Before: {len(df)} rows (8 snapshots × {len(df)//8} cycles)")
    print(f"    After:  {len(df_agg)} rows (1 per cycle)")
    
    return df_agg


def engineer_ww_features(df):
    """
    Pipeline completa feature engineering.
    
    FLUSSO:
    -------
    Input: DataFrame raw dal CSV
      ↓
    [create_base_features()]
      → Crea 6 feature ingegnerizzate da sensori grezzi
      → Ogni snapshot ora ha: 15 sensori + 6 feature = 21 colonne
      ↓
    [aggregate_by_cycle()]
      → 8 snapshot → 1 valore per ciclo (media)
      → Riduce righe da N×8 a N
      ↓
    [clean()]
      → Rimuovi valori invalidi (NaN, inf)
      → Drop righe problematiche
      ↓
    Output: DataFrame pulito pronto per LSTM
    
    STATISTICHE STAMPATE:
    ---------------------
    - Shape input/output
    - Numero ESN (motori)
    - Cicli per ESN (media)
    - NaN trovati (se presenti)
    
    INPUT:
    ------
    df: DataFrame raw con tutti gli 8 snapshot per ciclo
    
    OUTPUT:
    -------
    df_clean: DataFrame processato con:
              - 1 riga per ciclo
              - 15 sensori raw + 6 feature = 21 colonne feature
              - No NaN/inf
              - Ordinato per ESN, Cycles
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    # Statistiche input
    print(f"\nInput shape: {df.shape}")
    print(f"  Total rows: {df.shape[0]:,}")
    print(f"  ESNs: {df['ESN'].nunique()}")
    print(f"  Unique cycles: {df.groupby('ESN')['Cycles'].nunique().sum():,}")
    print(f"  Snapshots per cycle: {df['Snapshot'].nunique()}")
    
    # Verifica 8 snapshot
    snapshots_per_cycle = df.groupby(['ESN', 'Cycles']).size().unique()
    if len(snapshots_per_cycle) == 1 and snapshots_per_cycle[0] == 8:
        print("  ✅ Confirmed: 8 snapshots per cycle")
    else:
        print(f"  ⚠️  Warning: inconsistent snapshots per cycle: {snapshots_per_cycle}")
    
    # Step 1: Crea feature
    df = create_base_features(df)
    
    # Step 2: Aggrega 8 snapshot → 1 per ciclo
    df_agg = aggregate_by_cycle(df)
    
    # Statistiche post-aggregazione
    print(f"\nAfter aggregation: {df_agg.shape}")
    cycles_per_esn = df_agg.groupby('ESN').size()
    print(f"  Cycles per ESN:")
    for esn, count in cycles_per_esn.items():
        print(f"    ESN {esn}: {count:,} cycles")
    print(f"  Average: {cycles_per_esn.mean():.0f} cycles/ESN")
    
    # Step 3: Pulisci valori invalidi
    print("\n  Cleaning invalid values...")
    
    # Conta NaN prima
    nan_before = df_agg.isnull().sum().sum()
    if nan_before > 0:
        print(f"    Found {nan_before} NaN values")
        nan_counts = df_agg.isnull().sum()
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"      {col}: {count}")
    
    # Sostituisci inf con NaN
    df_clean = df_agg.replace([np.inf, -np.inf], np.nan)
    
    # Conta inf sostituiti
    inf_count = df_clean.isnull().sum().sum() - nan_before
    if inf_count > 0:
        print(f"    Replaced {inf_count} inf values with NaN")
    
    # Drop NaN
    rows_before = len(df_clean)
    df_clean = df_clean.dropna()
    rows_dropped = rows_before - len(df_clean)
    
    if rows_dropped > 0:
        print(f"    Dropped {rows_dropped} rows with NaN/inf")
    else:
        print(f"    ✅ No invalid values found")
    
    # Statistiche finali
    print(f"\nFinal shape: {df_clean.shape}")
    print(f"  Rows: {len(df_clean):,}")
    print(f"  Features: {df_clean.shape[1] - 3}")  # -3 per ESN, Cycles, target
    
    return df_clean


# ============================================================================
# SEZIONE 2: SEQUENCE CREATION
# ============================================================================

def create_sequences(df, feature_cols, target_col, window_size):
    """
    Trasforma DataFrame → sequenze temporali per LSTM.
    
    CONCETTO CHIAVE: SLIDING WINDOW
    --------------------------------
    LSTM non lavora su punti singoli, ma su SEQUENZE temporali.
    Questa funzione crea sequenze usando una "finestra scorrevole".
    
    ESEMPIO VISIVO (window=30):
    ---------------------------
    Cicli disponibili: [0, 1, 2, ..., 999]  (1000 cicli totali)
    
    Sequenza 0:
      Window: [0, 1, 2, ..., 29]  (30 cicli)
      Input:  matrice 30×N_features (valori dei 30 cicli)
      Target: RUL al Cycle 29
    
    Sequenza 1:
      Window: [1, 2, 3, ..., 30]  (30 cicli, spostato di 1)
      Input:  matrice 30×N_features
      Target: RUL al Cycle 30
    
    Sequenza 2:
      Window: [2, 3, 4, ..., 31]
      Input:  matrice 30×N_features
      Target: RUL al Cycle 31
    
    ...
    
    Sequenza 970:
      Window: [970, 971, ..., 999]
      Input:  matrice 30×N_features
      Target: RUL al Cycle 999
    
    Totale sequenze create: 1000 - 30 + 1 = 971
    
    PERCHÉ FUNZIONA:
    ----------------
    LSTM vede STORIA (ultimi K cicli) e impara pattern tipo:
    - "Feature sta salendo linearmente → RUL alto"
    - "Feature ha spike improvviso → RUL basso (failure vicina)"
    - "Temperatura sale + velocità sale → usura accelerata"
    
    MATEMATICA:
    -----------
    N_cycles = numero cicli disponibili
    window_size = lunghezza finestra
    N_sequences = N_cycles - window_size + 1
    
    SHAPE OUTPUT:
    -------------
    X_seq: (N_sequences, window_size, N_features)
           Esempio: (7970, 30, 21) = 7970 sequenze, 30 timesteps, 21 features
    
    y_seq: (N_sequences,)
           Esempio: (7970,) = 7970 valori target
    
    INPUT:
    ------
    df: DataFrame con 1 riga per ciclo, ordinato per (ESN, Cycles)
    feature_cols: lista nomi colonne da usare come input (es. 21 features)
    target_col: nome colonna target (es. 'Cycles_to_WW')
    window_size: lunghezza finestra temporale (es. 30)
    
    OUTPUT:
    -------
    X_seq: array 3D (N_sequences, window_size, N_features)
    y_seq: array 1D (N_sequences,)
    info: DataFrame con metadata (ESN, Cycle per ogni sequenza)
    """
    print(f"\n  Creating sequences (window={window_size})...")
    
    X_seq = []  # Lista sequenze input
    y_seq = []  # Lista target
    cycle_info = []  # Metadata
    
    # Processa ogni motore separatamente (non mischiare ESN diversi)
    for esn in sorted(df['ESN'].unique()):
        # Filtra e ordina
        d_esn = df[df['ESN'] == esn].sort_values('Cycles').reset_index(drop=True)
        
        # Estrai feature e target come array numpy
        X = d_esn[feature_cols].values  # Shape: (N_cycles, N_features)
        y = d_esn[target_col].values     # Shape: (N_cycles,)
        cycles = d_esn['Cycles'].values  # Per metadata
        
        # Sliding window: scorri lungo i cicli
        n_sequences_esn = len(X) - window_size + 1
        
        for i in range(n_sequences_esn):
            # Estrai finestra [i, i+window_size)
            X_window = X[i:i+window_size]  # Shape: (window_size, N_features)
            
            # Target = RUL all'ULTIMO ciclo della finestra
            y_target = y[i+window_size-1]
            
            # Salva
            X_seq.append(X_window)
            y_seq.append(y_target)
            
            # Metadata per debug/visualizzazione
            cycle_info.append({
                'ESN': esn,
                'Cycle_end': cycles[i+window_size-1],
                'Cycle_start': cycles[i],
                'Sequence_idx': len(X_seq) - 1
            })
        
        print(f"    ESN {esn}: {len(X)} cycles → {n_sequences_esn} sequences")
    
    # Converti liste → array numpy
    X_seq = np.array(X_seq)  # (N_sequences, window_size, N_features)
    y_seq = np.array(y_seq)  # (N_sequences,)
    info = pd.DataFrame(cycle_info)
    
    print(f"\n  Total sequences created: {X_seq.shape[0]:,}")
    print(f"  Input shape:  {X_seq.shape}")
    print(f"  Target shape: {y_seq.shape}")
    print(f"  Target range: [{y_seq.min():.1f}, {y_seq.max():.1f}]")
    
    return X_seq, y_seq, info


# ============================================================================
# SEZIONE 3: MODEL ARCHITECTURE
# ============================================================================

class AttentionLayer(layers.Layer):
    """
    Attention mechanism custom per Keras.
    
    IDEA CENTRALE:
    --------------
    Non tutti i timesteps (cicli) nella sequenza sono ugualmente importanti.
    Alcuni cicli contengono segnali critici (es. spike anomalo), altri
    sono rumore di fondo.
    
    Attention IMPARA automaticamente quali timesteps meritano più "attenzione".
    
    ESEMPIO PRATICO:
    ----------------
    Sequenza di 30 cicli: [Cycle 970, 971, ..., 999]
    
    Senza attention:
      Ogni ciclo pesa 1/30 = 3.33%
      Output = media semplice di tutti i 30 cicli
    
    Con attention (esempio pesi appresi):
      Cycle 970: peso 1%   (lontano, baseline normale)
      Cycle 975: peso 2%   (normale)
      Cycle 980: peso 3%   (leggero aumento temperatura)
      Cycle 985: peso 5%   (temperatura continua a salire)
      Cycle 990: peso 30%  ← SPIKE ANOMALO! Alta importanza
      Cycle 995: peso 15%  (post-spike, ancora rilevante)
      Cycle 999: peso 10%  (ciclo finale, importante)
      ...resto: 34% distribuito
    
    Output = somma PESATA, dominata dal Cycle 990
    
    La rete ha IMPARATO che spike improvvisi predicono failure imminente!
    
    MATEMATICA:
    -----------
    Input:  x con shape (batch, timesteps, features)
    
    Step 1: Calcola score di "importanza"
      e = tanh(x · W + b)
      W, b sono parametri TRAINABLE (la rete li impara)
    
    Step 2: Normalizza con softmax
      a = softmax(e)
      Garantisce: Σ(a) = 1 (somma pesi = 100%)
    
    Step 3: Somma pesata
      output = Σ(x * a)
    
    Output: (batch, features)
    
    ANALOGIA:
    ---------
    Immagina un medico che analizza 30 giorni di temperatura corporea.
    Invece di fare media semplice, dà PIÙ PESO ai giorni con febbre alta
    (40°C) rispetto a giorni normali (36.5°C).
    
    Attention fa la stessa cosa: peso alto ai cicli "anomali/critici".
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape = (batch, timesteps, features)
        # Es. (None, 30, 21) → None=batch variable, 30 timesteps, 21 features
        
        # Weight matrix W per calcolare attention scores
        # Shape: (features, features) = (21, 21)
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',  # Xavier initialization
            trainable=True  # Questi pesi vengono IMPARATI durante training
        )
        
        # Bias vector b
        # Shape: (features,) = (21,)
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        """
        Forward pass dell'attention layer.
        
        x shape: (batch, timesteps, features) = (32, 30, 21)
        """
        
        # Step 1: Calcola score
        # tensordot(x, W) = matrix multiplication along last axis
        # Shape: (batch, timesteps, features) @ (features, features)
        #      = (batch, timesteps, features)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        
        # Step 2: Normalizza con softmax
        # axis=1 → softmax lungo timesteps dimension
        # Garantisce: per ogni (batch, feature), Σ_timesteps(a) = 1
        a = tf.nn.softmax(e, axis=1)
        # a shape: (batch, timesteps, features)
        
        # Step 3: Somma pesata (element-wise multiplication + sum)
        output = x * a  # Broadcasting: (batch, timesteps, features)
        output = tf.reduce_sum(output, axis=1)  # Sum along timesteps
        # output shape: (batch, features) = (32, 21)
        
        return output
    
    def compute_output_shape(self, input_shape):
        # (batch, timesteps, features) → (batch, features)
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        """Necessario per salvare/caricare modello."""
        config = super().get_config()
        return config


def build_lstm_attention_model(timesteps, n_features, 
                                lstm_units=[64, 32], 
                                dense_units=[16], 
                                dropout=0.3, 
                                l2_reg=0.001):
    """
    Costruisce modello LSTM + Attention per predizione RUL.
    
    ARCHITETTURA LAYER-BY-LAYER:
    -----------------------------
    
    Layer 0: INPUT
      Shape: (batch, timesteps, n_features)
      Esempio: (32, 30, 21)
        32 = batch size (numero sequenze processate insieme)
        30 = timesteps (lunghezza finestra temporale)
        21 = features (sensori + engineered features)
    
    ↓
    
    Layer 1: LSTM (64 units, return_sequences=True)
      Tipo: Recurrent Neural Network
      Funzione: Impara pattern temporali BASE
      Esempi pattern:
        - "Feature X sta aumentando linearmente"
        - "Feature Y oscilla con periodo di 10 cicli"
        - "Feature Z ha spike ogni 50 cicli"
      Output shape: (batch, timesteps, 64)
        Mantiene timesteps! Ogni timestep ora rappresentato da 64 neuroni
    
    ↓
    
    Layer 2: DROPOUT (30%)
      Tipo: Regularization
      Funzione: Durante training, "spegne" casualmente 30% neuroni
      Perché: Previene overfitting (memorizzazione dataset)
      Output shape: (batch, timesteps, 64) [invariato]
    
    ↓
    
    Layer 3: LSTM (32 units, return_sequences=True)
      Funzione: Affina pattern appresi da LSTM1
      Esempi pattern:
        - "Quando X sale E Y scende → failure vicina"
        - "Spike in Z seguiti da plateau in W → usura accelerata"
      Output shape: (batch, timesteps, 32)
    
    ↓
    
    Layer 4: DROPOUT (30%)
      Output shape: (batch, timesteps, 32)
    
    ↓
    
    Layer 5: ATTENTION
      Tipo: Custom layer (definita sopra)
      Funzione: Focalizza su timesteps IMPORTANTI
      Meccanismo: Impara pesi per ogni timestep (somma=1)
      Output shape: (batch, 32)
        ⚠️ COLLASSA timesteps! Da (batch, 30, 32) → (batch, 32)
        Usa somma pesata: output = Σ(timestep_i × weight_i)
    
    ↓
    
    Layer 6: DENSE (16 units, ReLU)
      Tipo: Fully Connected
      Funzione: Combina 32 feature da attention in 16 "high-level features"
      Activation: ReLU(x) = max(0, x) [non-linearità]
      Output shape: (batch, 16)
    
    ↓
    
    Layer 7: DROPOUT (30%)
      Output shape: (batch, 16)
    
    ↓
    
    Layer 8: OUTPUT (1 unit, Linear)
      Tipo: Fully Connected
      Funzione: Predizione finale
      Activation: Linear (no activation, output può essere qualsiasi valore)
      Output shape: (batch, 1)
        Questo è log(RUL+1), verrà invertito a RUL dopo predizione
    
    ═══════════════════════════════════════════════════════════════
    
    HYPERPARAMETERS:
    ----------------
    timesteps: lunghezza sequenza (es. 30 cicli)
    n_features: numero feature input (es. 21)
    lstm_units: [64, 32] = 2 LSTM layers con 64 e 32 neuroni
    dense_units: [16] = 1 Dense layer con 16 neuroni
    dropout: 0.3 = 30% dropout (previene overfitting)
    l2_reg: 0.001 = L2 regularization (penalizza pesi grandi)
    
    TOTAL PARAMETERS: ~50,000 parametri trainable
    
    TRAINING:
    ---------
    Optimizer: Adam (adaptive learning rate)
    Loss: MSE (Mean Squared Error) tra log(RUL) predetto e vero
    Metrics: MAE (Mean Absolute Error) per monitoring
    """
    
    print(f"\n  Building LSTM+Attention model...")
    print(f"    Input: ({timesteps} timesteps, {n_features} features)")
    print(f"    LSTM layers: {lstm_units}")
    print(f"    Dense layers: {dense_units}")
    print(f"    Dropout: {dropout*100:.0f}%")
    print(f"    L2 regularization: {l2_reg}")
    
    # Layer 0: Input
    inputs = keras.Input(shape=(timesteps, n_features), name='input')
    
    # Layers 1-4: Stacked LSTM + Dropout
    x = inputs
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(
            units,
            return_sequences=True,  # CRITICO! Mantieni timesteps
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            name=f'lstm_{i+1}'
        )(x)
        x = layers.Dropout(dropout, name=f'dropout_lstm_{i+1}')(x)
        print(f"    LSTM layer {i+1}: {units} units")
    
    # Layer 5: Attention (collassa timesteps)
    x = AttentionLayer(name='attention')(x)
    print(f"    Attention layer: collapse {timesteps} timesteps → 1")
    
    # Layers 6-7: Dense + Dropout
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        )(x)
        x = layers.Dropout(dropout, name=f'dropout_dense_{i+1}')(x)
        print(f"    Dense layer {i+1}: {units} units")
    
    # Layer 8: Output
    outputs = layers.Dense(1, activation='linear', name='output')(x)
    print(f"    Output layer: 1 unit (log-RUL prediction)")
    
    # Crea modello
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Attention_WW')
    
    # Compila
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Conta parametri
    total_params = model.count_params()
    print(f"\n    Total parameters: {total_params:,}")
    
    return model


# ============================================================================
# SEZIONE 4: TRAINING
# ============================================================================

def train_model_for_window(df, feature_cols, target_col, window_size):
    """
    Training completo per una data window size.
    
    PIPELINE COMPLETA:
    ------------------
    1. Crea sequenze temporali (sliding window)
    2. Log-transform target (stabilizza varianza)
    3. Standardizza features (mean=0, std=1)
    4. Split train/validation (80/20)
    5. Build model architecture
    6. Train con callbacks (early stopping, learning rate decay)
    7. Evaluate su tutti i dati
    8. Return risultati + oggetti salvati
    
    INPUT:
    ------
    df: DataFrame processato (1 riga/ciclo, no NaN)
    feature_cols: lista feature da usare
    target_col: nome colonna target
    window_size: lunghezza finestra temporale
    
    OUTPUT:
    -------
    dict con chiavi:
      'window': window_size usato
      'mae': Mean Absolute Error
      'r2': R² score
      'correlation': Pearson correlation
      'epochs': numero epoch training
      'n_sequences': numero sequenze create
      'model': modello Keras trainato
      'scaler': StandardScaler fittato
      'history': training history
      'X': sequenze input (scaled)
      'y': target (original scale)
      'y_pred': predizioni (original scale)
      'info': metadata sequenze
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: Window Size = {window_size} cycles")
    print('='*70)
    
    # Step 1: Crea sequenze
    X_seq, y_seq, info = create_sequences(
        df, feature_cols, target_col, window_size
    )
    
    if X_seq.shape[0] < 100:
        print(f"\n⚠️  Too few sequences ({X_seq.shape[0]}), skipping window={window_size}")
        return None
    
    # Step 2: Log-transform target
    print("\n  Transforming target to log-scale...")
    y_log = np.log(y_seq + 1)  # +1 evita log(0) = -inf
    
    print(f"    Original scale:  [{y_seq.min():.1f}, {y_seq.max():.1f}]")
    print(f"    Log scale:       [{y_log.min():.3f}, {y_log.max():.3f}]")
    print(f"    Why log? Stabilizes variance (high RUL → high variance)")
    
    # Step 3: Standardizza features
    print("\n  Standardizing features (mean=0, std=1)...")
    scaler = StandardScaler()
    
    # Reshape per StandardScaler: (N_seq, timesteps, features) → (N_seq*timesteps, features)
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    print(f"    Flattened shape: {X_flat.shape}")
    
    # Fit scaler e transform
    X_scaled_flat = scaler.fit_transform(X_flat)
    
    # Reshape back: (N_seq*timesteps, features) → (N_seq, timesteps, features)
    X_scaled = X_scaled_flat.reshape(X_seq.shape)
    print(f"    Reshaped back: {X_scaled.shape}")
    
    # Verifica standardizzazione
    print(f"    Mean after scaling: {X_scaled.mean():.6f} (should be ~0)")
    print(f"    Std after scaling:  {X_scaled.std():.6f} (should be ~1)")
    
    # Step 4: Train/validation split
    print("\n  Splitting train/validation (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"    Train: {X_train.shape[0]:,} sequences ({X_train.shape[0]/X_scaled.shape[0]*100:.1f}%)")
    print(f"    Val:   {X_val.shape[0]:,} sequences ({X_val.shape[0]/X_scaled.shape[0]*100:.1f}%)")
    
    # Step 5: Build model
    model = build_lstm_attention_model(
        timesteps=window_size,
        n_features=X_seq.shape[-1],
        lstm_units=[64, 32],
        dense_units=[16],
        dropout=0.3,
        l2_reg=0.001
    )
    
    # Step 6: Train
    print("\n  Training model...")
    print("    Max epochs: 300")
    print("    Batch size: 32")
    print("    Callbacks:")
    print("      - EarlyStopping: patience=30 (stop if no improvement)")
    print("      - ReduceLROnPlateau: patience=10 (halve LR if plateau)")
    
    # Callback 1: Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    
    # Callback 2: Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=2  # 1 line per epoch
    )
    
    print(f"\n    Training completed: {len(history.history['loss'])} epochs")
    
    # Step 7: Evaluate
    print("\n  Evaluating on all data...")
    
    # Predict in log-space
    y_pred_log = model.predict(X_scaled, verbose=0).flatten()
    
    # Inverse transform: log → original
    y_pred = np.exp(y_pred_log) - 1
    
    # Clip negative predictions (fisica: RUL non può essere negativo)
    y_pred = np.clip(y_pred, 0, None)
    n_clipped = (y_pred == 0).sum()
    if n_clipped > 0:
        print(f"    Clipped {n_clipped} negative predictions to 0")
    
    # Metrics
    mae = mean_absolute_error(y_seq, y_pred)
    r2 = r2_score(y_seq, y_pred)
    corr = np.sqrt(r2) if r2 > 0 else 0
    
    # Metrics per ESN
    print("\n    Metrics per ESN:")
    for esn in sorted(info['ESN'].unique()):
        mask = info['ESN'] == esn
        mae_esn = mean_absolute_error(y_seq[mask], y_pred[mask])
        r2_esn = r2_score(y_seq[mask], y_pred[mask])
        print(f"      ESN {esn}: MAE={mae_esn:6.1f}, R²={r2_esn:.3f}")
    
    print(f"\n  ✅ RESULTS (window={window_size}):")
    print(f"     MAE: {mae:.1f} cycles")
    print(f"     R²:  {r2:.3f}")
    print(f"     Correlation: {corr:.3f}")
    
    # Return tutto
    return {
        'window': window_size,
        'mae': mae,
        'r2': r2,
        'correlation': corr,
        'epochs': len(history.history['loss']),
        'n_sequences': X_seq.shape[0],
        'model': model,
        'scaler': scaler,
        'history': history,
        'X': X_scaled,
        'y': y_seq,
        'y_pred': y_pred,
        'info': info
    }


def train_all_windows(df, feature_cols, target_col, window_sizes):
    """
    Training per tutte le window sizes.
    
    Loop su ogni window size e chiama train_model_for_window().
    Raccoglie risultati in DataFrame + dict.
    
    INPUT:
    ------
    df: DataFrame processato
    feature_cols: lista feature
    target_col: nome target
    window_sizes: lista window sizes da testare (es. [10, 30, 50, 100])
    
    OUTPUT:
    -------
    results_df: DataFrame con 1 riga per window (summary)
    results_dict: dict con oggetti completi per ogni window
    """
    print("\n" + "="*70)
    print("TRAINING ALL WINDOW SIZES")
    print("="*70)
    print(f"Testing window sizes: {window_sizes}")
    
    results_list = []
    results_dict = {}
    
    for window in window_sizes:
        result = train_model_for_window(df, feature_cols, target_col, window)
        
        if result is not None:
            # Summary per DataFrame
            results_list.append({
                'Window': result['window'],
                'MAE': result['mae'],
                'R2': result['r2'],
                'Correlation': result['correlation'],
                'Epochs': result['epochs'],
                'N_sequences': result['n_sequences']
            })
            
            # Oggetti completi
            results_dict[window] = result
        
        print()  # Blank line tra window sizes
    
    results_df = pd.DataFrame(results_list)
    
    return results_df, results_dict


# ============================================================================
# SEZIONE 5: VISUALIZATION
# ============================================================================

def plot_window_comparison(results_df, out_dir):
    """Plot MAE, R², Correlation vs window size."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: MAE
    ax = axes[0]
    ax.plot(results_df['Window'], results_df['MAE'], 
           'o-', linewidth=2.5, markersize=12, color='steelblue')
    
    best_idx = results_df['MAE'].idxmin()
    best_window = int(results_df.loc[best_idx, 'Window'])
    best_mae = results_df.loc[best_idx, 'MAE']
    ax.scatter([best_window], [best_mae], color='red', 
              s=250, zorder=5, edgecolors='darkred', linewidths=2,
              label=f'Best: window={best_window}')
    
    ax.set_xlabel('Window Size (cycles)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MAE (cycles)', fontsize=13, fontweight='bold')
    ax.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: R²
    ax = axes[1]
    ax.plot(results_df['Window'], results_df['R2'], 
           'o-', linewidth=2.5, markersize=12, color='forestgreen')
    
    best_r2_idx = results_df['R2'].idxmax()
    best_r2_window = int(results_df.loc[best_r2_idx, 'Window'])
    best_r2 = results_df.loc[best_r2_idx, 'R2']
    ax.scatter([best_r2_window], [best_r2], color='red', 
              s=250, zorder=5, edgecolors='darkred', linewidths=2,
              label=f'Best: window={best_r2_window}')
    
    ax.set_xlabel('Window Size (cycles)', fontsize=13, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax.set_title('R² Score (Explained Variance)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Correlation
    ax = axes[2]
    ax.plot(results_df['Window'], results_df['Correlation'], 
           'o-', linewidth=2.5, markersize=12, color='purple')
    
    best_corr_idx = results_df['Correlation'].idxmax()
    best_corr_window = int(results_df.loc[best_corr_idx, 'Window'])
    best_corr = results_df.loc[best_corr_idx, 'Correlation']
    ax.scatter([best_corr_window], [best_corr], color='red', 
              s=250, zorder=5, edgecolors='darkred', linewidths=2,
              label=f'Best: window={best_corr_window}')
    
    ax.set_xlabel('Window Size (cycles)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=13, fontweight='bold')
    ax.set_title('Pearson Correlation (√R²)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    fname = "window_comparison.png"
    fig.savefig(f"{out_dir}/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {fname}")


def plot_predictions_per_esn(result, out_dir):
    """Plot actual vs predicted per ESN (best window)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    info = result['info']
    y = result['y']
    y_pred = result['y_pred']
    window = result['window']
    
    esns = sorted(info['ESN'].unique())
    
    for i, esn in enumerate(esns):
        ax = axes[i]
        
        mask = info['ESN'] == esn
        y_esn = y[mask]
        y_pred_esn = y_pred[mask]
        
        # Scatter colored by sequence index
        scatter = ax.scatter(
            y_esn, y_pred_esn, 
            c=np.arange(len(y_esn)), 
            cmap='viridis', 
            s=25, 
            alpha=0.7, 
            edgecolors='none'
        )
        
        # Perfect prediction line (y=x)
        lims = [0, max(y_esn.max(), y_pred_esn.max()) * 1.05]
        ax.plot(lims, lims, 'r--', linewidth=2.5, alpha=0.8, 
               label='Perfect prediction', zorder=10)
        
        # Metrics
        mae = mean_absolute_error(y_esn, y_pred_esn)
        r2 = r2_score(y_esn, y_pred_esn)
        corr = np.corrcoef(y_esn, y_pred_esn)[0,1]
        
        textstr = f'MAE: {mae:.1f}\nR²: {r2:.3f}\nCorr: {corr:.3f}\nN: {len(y_esn):,}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.95)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', bbox=props,
               fontweight='bold')
        
        ax.set_xlabel('Actual RUL (cycles)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted RUL (cycles)', fontsize=12, fontweight='bold')
        ax.set_title(f'ESN {esn}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        if i == len(esns) - 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Sequence index (time →)', fontsize=10)
    
    plt.suptitle(
        f'LSTM+Attention Predictions | Window={window} cycles | '
        f'Overall MAE={result["mae"]:.1f}, R²={result["r2"]:.3f}', 
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fname = f"predictions_window{window}.png"
    fig.savefig(f"{out_dir}/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fname}")


# ============================================================================
# SEZIONE 6: MAIN
# ============================================================================

def main():
    """
    Main execution pipeline.
    
    FLUSSO COMPLETO:
    ----------------
    1. Load data from CSV
    2. Feature engineering (8 snapshot → 1 per ciclo)
    3. Define feature columns
    4. Train models per ogni window size
    5. Compare results
    6. Save best model + plots
    """
    
    print("\n" + "="*70)
    print("LSTM + ATTENTION FOR WW PREDICTION")
    print("Remaining Useful Life prediction using temporal sequences")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    cfg = load_config("configs/config.yaml")
    df = pd.read_csv(cfg["data"]["train_csv"])
    
    print(f"  Raw data shape: {df.shape}")
    print(f"  ESNs (engines): {df['ESN'].nunique()}")
    print(f"  Unique cycles: {df.groupby('ESN')['Cycles'].nunique().sum():,}")
    print(f"  Snapshots: {df['Snapshot'].nunique()}")
    
    # Feature engineering (TUTTI gli 8 snapshot)
    df_eng = engineer_ww_features(df)
    
    # Define feature columns
    # 15 raw sensors + 6 engineered = 21 features total
    feature_cols = [
        # Engineered features (6)
        'temp_diff_norm',      # Stress termico normalizzato
        'thermal_stress',      # Stress termico assoluto
        'load_factor',         # Carico motore
        'speed_ratio',         # Rapporto velocità core/fan
        'temp_gradient',       # Gradiente temperatura
        'pressure_ratio',      # Rapporto pressioni
        # Raw sensors (15)
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
        'Sensed_T45'
    ]
    
    print(f"\n{'='*70}")
    print(f"FEATURES: {len(feature_cols)} total")
    print(f"{'='*70}")
    print("\nEngineered (6):")
    for i, col in enumerate(feature_cols[:6], 1):
        print(f"  {i}. {col}")
    print("\nRaw Sensors (15):")
    for i, col in enumerate(feature_cols[6:], 1):
        print(f"  {i}. {col}")
    
    # Train all windows
    window_sizes = [10, 30, 50, 100]
    print(f"\n{'='*70}")
    print(f"WINDOW SIZES TO TEST: {window_sizes}")
    print(f"{'='*70}")
    
    results_df, results_dict = train_all_windows(
        df_eng, feature_cols, 'Cycles_to_WW', window_sizes
    )
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Best window
    best_idx = results_df['MAE'].idxmin()
    best_window = int(results_df.loc[best_idx, 'Window'])
    best_result = results_dict[best_window]
    
    print("\n" + "="*70)
    print("🏆 BEST CONFIGURATION")
    print("="*70)
    print(f"\nWindow Size: {best_window} cycles")
    print(f"  MAE:         {best_result['mae']:.2f} cycles")
    print(f"  R²:          {best_result['r2']:.4f}")
    print(f"  Correlation: {best_result['correlation']:.4f}")
    print(f"  Epochs:      {best_result['epochs']}")
    print(f"  Sequences:   {best_result['n_sequences']:,}")
    
    # Comparison with baseline
    print(f"\nImprovement over baseline (corr=0.118):")
    baseline_corr = 0.118
    improvement = ((best_result['correlation'] - baseline_corr) / baseline_corr) * 100
    print(f"  Correlation: {baseline_corr:.3f} → {best_result['correlation']:.3f}")
    print(f"  Improvement: +{improvement:.1f}%")
    print(f"  Factor: {best_result['correlation']/baseline_corr:.1f}× better")
    
    # Save results
    out_dir = "artifacts/ww_lstm_final"
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # CSV
    results_df.to_csv(f"{out_dir}/window_comparison.csv", index=False)
    print(f"\n✅ window_comparison.csv")
    
    # Plots
    plot_window_comparison(results_df, out_dir)
    plot_predictions_per_esn(best_result, out_dir)
    
    # Model
    model_path = f"{out_dir}/lstm_model_window{best_window}.h5"
    best_result['model'].save(model_path)
    print(f"✅ lstm_model_window{best_window}.h5")
    
    # Scaler (per predizioni su test set)
    import joblib
    scaler_path = f"{out_dir}/scaler_window{best_window}.pkl"
    joblib.dump(best_result['scaler'], scaler_path)
    print(f"✅ scaler_window{best_window}.pkl")
    
    print(f"\n📁 All results saved in: {out_dir}/")
    
    # Final assessment
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)
    
    if best_result['mae'] < 100:
        print("\n🎉 EXCELLENT: MAE < 100 cycles!")
        print("   Model is production-ready for WW prediction")
    elif best_result['mae'] < 150:
        print("\n✅ VERY GOOD: MAE < 150 cycles!")
        print("   Model performs well, suitable for most applications")
    elif best_result['mae'] < 200:
        print("\n✅ GOOD: MAE < 200 cycles!")
        print("   Model shows significant improvement over baseline")
    else:
        print("\n⚠️  MODERATE: MAE still above 200 cycles")
        print("   Consider: more data, different architecture, or domain features")
    
    print("\n" + "="*70)
    print("DONE ✨")
    print("="*70)


if __name__ == "__main__":
    main()
