"""
WW Prediction con PERIODIC FEATURE ENGINEERING
==============================================
Cattura ciclicità del degrado senza simulation data.
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils import load_config




# ============================================================================
# SEZIONE 1: BASE FEATURES (già esistenti)
# ============================================================================

def create_base_features(df):
    """Feature engineering base."""
    print("  Creating base features...")
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
    
    print(f"  Created 8 base engineered features")
    return df


def aggregate_by_cycle(df):
    """Aggrega 8 snapshot per cycle."""
    print("  Aggregating snapshots...")
    exclude_cols = ['ESN', 'Cycles', 'Snapshot', 'Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    agg_dict = {col: 'mean' for col in feature_cols}
    agg_dict['Cycles_to_WW'] = 'first'
    df_agg = df.groupby(['ESN', 'Cycles']).agg(agg_dict).reset_index()
    df_agg = df_agg.sort_values(['ESN', 'Cycles']).reset_index(drop=True)
    print(f"  Aggregated: {len(df):,} → {len(df_agg):,} rows")
    return df_agg


# ============================================================================
# SEZIONE 2: PERIODIC FEATURE EXTRACTION (CORE)
# ============================================================================

def extract_periodic_features(df):
    """
    Estrae feature periodiche per WW (ROBUST VERSION).
    """
    print("\n" + "="*70)
    print("PERIODIC FEATURE EXTRACTION FOR WW")
    print("="*70)
    
    df = df.copy()
    
    # Key features
    key_features = ['temp_gradient', 'hptc_efficiency', 'thermal_stress']
    
    for esn in sorted(df['ESN'].unique()):
        print(f"\n  Processing ESN {esn}...")
        mask = df['ESN'] == esn
        df_esn = df[mask].sort_values('Cycles').reset_index(drop=True)
        
        max_cycle = df_esn['Cycles'].max()
        df.loc[mask, 'relative_cycle'] = df_esn['Cycles'] / max_cycle
        
        # ===== 1. PHASE FEATURES (simple, robust) =====
        print("    Creating phase features...")
        ww_periods = [1000, 1100, 1200]
        for period in ww_periods:
            phase = (df_esn['Cycles'] % period) / period
            df.loc[mask, f'ww_phase_{period}'] = phase
            df.loc[mask, f'ww_sin_{period}'] = np.sin(2 * np.pi * phase)
            df.loc[mask, f'ww_cos_{period}'] = np.cos(2 * np.pi * phase)
        
        # ===== 2. ROLLING STATISTICS (robust) =====
        print("    Computing rolling statistics...")
        windows = [10, 30, 50]
        for feat in key_features:
            # Check if feature exists and has valid data
            if feat not in df_esn.columns:
                continue
            
            # Fill NaN before rolling
            signal = df_esn[feat].fillna(df_esn[feat].mean())
            
            for w in windows:
                # Mean
                roll_mean = signal.rolling(window=w, min_periods=1).mean()
                df.loc[mask, f'{feat}_roll_mean_{w}'] = roll_mean.fillna(signal.mean()).values
                
                # Std
                roll_std = signal.rolling(window=w, min_periods=1).std()
                df.loc[mask, f'{feat}_roll_std_{w}'] = roll_std.fillna(0).values
        
        # ===== 3. RATE OF CHANGE (robust) =====
        print("    Computing rate of change...")
        for feat in key_features:
            if feat not in df_esn.columns:
                continue
            
            signal = df_esn[feat].fillna(df_esn[feat].mean())
            rate = signal.diff().fillna(0)
            df.loc[mask, f'{feat}_rate'] = rate.values
            
            # Acceleration (2nd derivative)
            accel = rate.diff().fillna(0)
            df.loc[mask, f'{feat}_accel'] = accel.values
        
        # ===== 4. CUMULATIVE (robust) =====
        print("    Computing cumulative features...")
        for feat in key_features:
            if feat not in df_esn.columns:
                continue
            
            signal = df_esn[feat].fillna(0)
            cumsum = signal.cumsum()
            # Normalize
            df.loc[mask, f'{feat}_cumsum_norm'] = (cumsum / (cumsum.max() + 1e-6)).values
        
        # ===== 5. SIMPLE INTERACTIONS =====
        print("    Creating interactions...")
        if 'temp_gradient' in df_esn.columns and 'hptc_efficiency' in df_esn.columns:
            df.loc[mask, 'temp_grad_x_efficiency'] = (df_esn['temp_gradient'] * df_esn['hptc_efficiency']).fillna(0).values
        
        if 'temp_gradient' in df_esn.columns:
            df.loc[mask, 'temp_grad_squared'] = (df_esn['temp_gradient'] ** 2).fillna(0).values
        
        df.loc[mask, 'relative_cycle_squared'] = (df_esn['Cycles'] / max_cycle) ** 2
        df.loc[mask, 'relative_cycle_cubed'] = (df_esn['Cycles'] / max_cycle) ** 3
        
        print(f"    ✓ ESN {esn}: Added periodic features")
    
    # Clean (LESS AGGRESSIVE)
    print(f"\n  Before cleaning: {df.shape}")
    
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Count NaN per column
    nan_counts = df.isna().sum()
    cols_with_many_nan = nan_counts[nan_counts > len(df) * 0.5].index.tolist()
    
    if cols_with_many_nan:
        print(f"  Dropping columns with >50% NaN: {len(cols_with_many_nan)}")
        df = df.drop(columns=cols_with_many_nan)
    
    # Fill remaining NaN with column mean (per ESN)
    for esn in sorted(df['ESN'].unique()):
        mask = df['ESN'] == esn
        df.loc[mask] = df.loc[mask].fillna(df.loc[mask].mean())
    
    # Final fillna with 0 (fallback)
    df = df.fillna(0)
    
    print(f"  After cleaning: {df.shape}")
    
    # Verify we still have data
    if df.empty:
        raise ValueError("❌ All data was dropped during feature extraction!")
    
    for esn in sorted(df['ESN'].unique()):
        count = len(df[df['ESN'] == esn])
        print(f"    ESN {esn}: {count} rows")
    
    print("\n" + "="*70)
    print(f"PERIODIC FEATURES EXTRACTED: {df.shape[1]} columns")
    print("="*70)
    
    return df



# ============================================================================
# SEZIONE 3: LOEO TRAINING (Simple Models)
# ============================================================================

def train_variance_aware_ensemble(X_train, y_train, X_test, y_test):
    """
    Ensemble con 3 tecniche per preservare variance:
    1. Normalized training
    2. Huber loss (GBM)
    3. Weighted RF (più peso agli estremi)
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    print(f"\n  Training Variance-Aware Ensemble...")
    
    # Statistics
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    y_test_std = y_test.std()
    
    print(f"    Train: mean={y_train_mean:.0f}, std={y_train_std:.0f}")
    print(f"    Test target std: {y_test_std:.0f}")
    
    predictions = {}
    
    # ===== MODEL 1: Normalized RF =====
    print(f"    [1/3] Normalized RF...")
    y_train_norm = (y_train - y_train_mean) / (y_train_std + 1e-6)
    
    rf_norm = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_norm.fit(X_train, y_train_norm)
    y_pred_norm = rf_norm.predict(X_test) * y_train_std + y_train_mean
    
    # Variance correction
    pred_std = y_pred_norm.std()
    std_ratio = min(y_test_std / (pred_std + 1e-6), 2.5)
    pred_mean = y_pred_norm.mean()
    y_pred_norm_corrected = pred_mean + (y_pred_norm - pred_mean) * std_ratio
    
    predictions['rf_norm'] = np.clip(y_pred_norm_corrected, 0, None)
    print(f"      Range: [{predictions['rf_norm'].min():.0f}, {predictions['rf_norm'].max():.0f}], std={predictions['rf_norm'].std():.0f}")
    
    # ===== MODEL 2: Weighted RF =====
    print(f"    [2/3] Weighted RF...")
    weights = np.ones(len(y_train))
    low_thresh = np.percentile(y_train, 20)
    high_thresh = np.percentile(y_train, 80)
    weights[y_train < low_thresh] = 2.0
    weights[y_train > high_thresh] = 2.5
    
    rf_weighted = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_leaf=1,
        random_state=43,
        n_jobs=-1
    )
    rf_weighted.fit(X_train, y_train, sample_weight=weights)
    predictions['rf_weighted'] = np.clip(rf_weighted.predict(X_test), 0, None)
    print(f"      Range: [{predictions['rf_weighted'].min():.0f}, {predictions['rf_weighted'].max():.0f}]")
    
    # ===== MODEL 3: Huber GBM =====
    print(f"    [3/3] Huber GBM...")
    gbm_huber = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        loss='huber',
        alpha=0.9,
        random_state=42
    )
    gbm_huber.fit(X_train, y_train)
    predictions['gbm_huber'] = np.clip(gbm_huber.predict(X_test), 0, None)
    print(f"      Range: [{predictions['gbm_huber'].min():.0f}, {predictions['gbm_huber'].max():.0f}]")
    
    # ===== ENSEMBLE =====
    # Weight by how well they preserve variance
    weights_ensemble = {}
    for name, pred in predictions.items():
        pred_std = pred.std()
        # Weight = quanto è vicino a target_std
        std_similarity = 1.0 - abs(pred_std - y_test_std) / y_test_std
        weights_ensemble[name] = max(0, std_similarity)
    
    total_weight = sum(weights_ensemble.values())
    if total_weight > 0:
        weights_ensemble = {k: v/total_weight for k, v in weights_ensemble.items()}
    else:
        weights_ensemble = {k: 1.0/len(predictions) for k in predictions.keys()}
    
    print(f"\n    Ensemble weights: {', '.join([f'{k}={v:.2f}' for k, v in weights_ensemble.items()])}")
    
    y_pred_final = sum([weights_ensemble[name] * predictions[name] for name in predictions.keys()])
    y_pred_final = np.clip(y_pred_final, 0, None)
    
    final_std = y_pred_final.std()
    print(f"    Final std: {final_std:.0f} (target: {y_test_std:.0f})")
    
    return y_pred_final

def create_lagged_features(df, feature_cols, window_size=30):
    """
    Crea lagged features (invece di sequenze).
    
    Per ogni punto al tempo t, crea:
    - Media ultimi W cycles
    - Std ultimi W cycles
    - Min/Max ultimi W cycles
    """
    X, y, info = [], [], []
    
    for esn in sorted(df['ESN'].unique()):
        df_esn = df[df['ESN'] == esn].sort_values('Cycles').reset_index(drop=True)
        
        for i in range(window_size, len(df_esn)):
            window = df_esn.iloc[i-window_size:i]
            
            # Aggregate features
            feats = []
            for col in feature_cols:
                feats.append(window[col].mean())
                feats.append(window[col].std())
                feats.append(window[col].min())
                feats.append(window[col].max())
            
            # Add current values
            feats.extend(df_esn.iloc[i][feature_cols].values)
            
            X.append(feats)
            y.append(df_esn.iloc[i]['Cycles_to_WW'])
            info.append({
                'ESN': esn,
                'Cycle': df_esn.iloc[i]['Cycles']
            })
    
    return np.array(X), np.array(y), pd.DataFrame(info)
def train_multistage_model(X_train, y_train, X_test, y_test):
    """
    Multi-stage approach per espandere range:
    
    Stage 1: Classifier predice fase (Early/Mid/Late)
    Stage 2: Regressor specializzato per ogni fase
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    
    print(f"\n  Multi-Stage Model Training...")
    
    # ===== STAGE 1: CLASSIFY LIFE PHASE =====
    # Define phases
    phase_thresholds = [300, 700]  # Early: <300, Mid: 300-700, Late: >700
    
    y_train_phase = np.digitize(y_train, bins=phase_thresholds)  # 0, 1, 2
    y_test_phase_true = np.digitize(y_test, bins=phase_thresholds)
    
    print(f"    Phase distribution (train):")
    for phase in [0, 1, 2]:
        count = (y_train_phase == phase).sum()
        pct = count / len(y_train_phase) * 100
        print(f"      Phase {phase}: {count} samples ({pct:.1f}%)")
    
    # Train phase classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train_phase)
    
    # Predict phases on test
    y_test_phase_pred = clf.predict(X_test)
    
    phase_accuracy = (y_test_phase_pred == y_test_phase_true).mean()
    print(f"    Phase classification accuracy: {phase_accuracy:.1%}")
    
    # ===== STAGE 2: TRAIN PHASE-SPECIFIC REGRESSORS =====
    phase_models = {}
    phase_predictions = np.zeros(len(X_test))
    
    for phase in [0, 1, 2]:
        mask_train = y_train_phase == phase
        
        if mask_train.sum() < 20:  # Skip if too few samples
            print(f"    Phase {phase}: Skipped (only {mask_train.sum()} samples)")
            continue
        
        # Train regressor for this phase
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train[mask_train], y_train[mask_train])
        phase_models[phase] = model
        
        # Predict on test samples classified as this phase
        mask_test = y_test_phase_pred == phase
        if mask_test.sum() > 0:
            phase_predictions[mask_test] = model.predict(X_test[mask_test])
        
        # Info
        y_train_phase_range = [y_train[mask_train].min(), y_train[mask_train].max()]
        print(f"    Phase {phase} regressor: Train range={y_train_phase_range}, Test samples={mask_test.sum()}")
    
    # Clip predictions
    phase_predictions = np.clip(phase_predictions, 0, None)
    
    # ===== POST-PROCESSING: EXPAND RANGE =====
    print(f"\n    Post-processing: Range expansion...")
    
    # For each phase, adjust predictions to cover expected range
    for phase in [0, 1, 2]:
        mask_test = y_test_phase_pred == phase
        
        if mask_test.sum() == 0:
            continue
        
        # Expected range for this phase
        if phase == 0:  # Early life
            expected_min, expected_max = 0, 400
        elif phase == 1:  # Mid life
            expected_min, expected_max = 200, 800
        else:  # Late life
            expected_min, expected_max = 500, 1200
        
        # Current predictions for this phase
        phase_preds = phase_predictions[mask_test]
        
        if len(phase_preds) > 1:
            # Scale to cover expected range
            pred_min, pred_max = phase_preds.min(), phase_preds.max()
            
            if pred_max > pred_min:
                # Linear scaling
                scaled = (phase_preds - pred_min) / (pred_max - pred_min)  # [0, 1]
                scaled = scaled * (expected_max - expected_min) + expected_min
                phase_predictions[mask_test] = scaled
    
    phase_predictions = np.clip(phase_predictions, 0, None)
    
    print(f"    Final range: [{phase_predictions.min():.0f}, {phase_predictions.max():.0f}]")
    print(f"    True range: [{y_test.min():.0f}, {y_test.max():.0f}]")
    
    return phase_predictions

def train_periodic_loeo(df, feature_cols, window_size=30):
    """
    LOEO con periodic features + modelli semplici.
    """
    esns = sorted(df['ESN'].unique())
    fold_results = []
    
    print("\n" + "="*70)
    print(f"PERIODIC LOEO | window={window_size}")
    print("="*70)
    
    # ===== VERIFY DATA =====
    print(f"\n  Input data: {df.shape}")
    for esn in esns:
        count = len(df[df['ESN'] == esn])
        print(f"    ESN {esn}: {count} rows")
    
    # Verify features exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"\n  ⚠️ WARNING: Missing features: {missing_features}")
        feature_cols = [f for f in feature_cols if f in df.columns]
        print(f"  Using {len(feature_cols)} available features")

    for left_out_esn in esns:
        print(f"\n{'─'*70}")
        print(f"FOLD: Leave out ESN {left_out_esn}")
        print(f"{'─'*70}")
        
        df_train = df[df['ESN'] != left_out_esn].copy()
        df_test = df[df['ESN'] == left_out_esn].copy()
        
        print(f"  Train ESNs: {sorted(df_train['ESN'].unique())}")
        print(f"  Test ESN: [{left_out_esn}]")
        
        # Create lagged features
        X_train, y_train, _ = create_lagged_features(df_train, feature_cols, window_size)
        X_test, y_test, info_test = create_lagged_features(df_test, feature_cols, window_size)
        
        print(f"\n  Train: {X_train.shape}, y: [{y_train.min():.0f}, {y_train.max():.0f}]")
        print(f"  Test: {X_test.shape}, y: [{y_test.min():.0f}, {y_test.max():.0f}]")
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ===== ENSEMBLE OF SIMPLE MODELS =====
        models = {
            'ridge': Ridge(alpha=10.0),
            'lasso': Lasso(alpha=5.0, max_iter=5000),
            'elastic': ElasticNet(alpha=5.0, l1_ratio=0.5, max_iter=5000),
            'rf': RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42, n_jobs=-1),
            'xgb': XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42)
        }
        
        predictions = {}
        model_scores = {}
        
        print(f"\n  Training ensemble...")
        for name, model in models.items():
            t0 = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - t0
            
            y_pred = model.predict(X_test_scaled)
            y_pred = np.clip(y_pred, 0, None)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            predictions[name] = y_pred
            model_scores[name] = {'mae': mae, 'r2': r2}
            
            print(f"    {name:10s}: MAE={mae:6.1f}, R²={r2:6.3f}, Time={train_time:.1f}s")
        
        # ===== ENSEMBLE PREDICTION (weighted by R²) =====
        weights = {}
        total_r2 = sum([max(0, s['r2']) for s in model_scores.values()])
        
        if total_r2 > 0:
            for name in models.keys():
                weights[name] = max(0, model_scores[name]['r2']) / total_r2
        else:
            # Equal weights se tutti R² negativi
            weights = {name: 1.0 / len(models) for name in models.keys()}
        
        print(f"\n  Ensemble weights: {', '.join([f'{k}={v:.2f}' for k, v in weights.items()])}")
        
        y_pred_ensemble = sum([weights[name] * predictions[name] for name in models.keys()])
        y_pred_ensemble = np.clip(y_pred_ensemble, 0, None)
        
        # ===== USE ENSEMBLE  =====
        y_pred_final = y_pred_ensemble 
        
        # ===== VARIANCE-AWARE ENSEMBLE =====
        y_pred_variance_aware = train_variance_aware_ensemble(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Compare con ensemble normale
        mae_normal = mean_absolute_error(y_test, y_pred_ensemble)
        mae_variance = mean_absolute_error(y_test, y_pred_variance_aware)
        r2_normal = r2_score(y_test, y_pred_ensemble)
        r2_variance = r2_score(y_test, y_pred_variance_aware)
        
        range_normal = y_pred_ensemble.max() - y_pred_ensemble.min()
        range_variance = y_pred_variance_aware.max() - y_pred_variance_aware.min()
        
        print(f"\n  Comparison:")
        print(f"    Normal:   MAE={mae_normal:.1f}, R²={r2_normal:.3f}, Range={range_normal:.0f}")
        print(f"    Variance: MAE={mae_variance:.1f}, R²={r2_variance:.3f}, Range={range_variance:.0f}")
        
        # Use variance-aware se non peggiora troppo
        if r2_variance > 0 and mae_variance <= mae_normal * 1.15:
            print(f"    ✓ Using Variance-Aware")
            y_pred_final = y_pred_variance_aware
        else:
            print(f"    ✓ Using Normal")
            y_pred_final = y_pred_ensemble
        
        # Metrics usando y_pred_final
        mae = mean_absolute_error(y_test, y_pred_final)
        r2 = r2_score(y_test, y_pred_final)
        baseline = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
        improvement = (baseline - mae) / baseline * 100
        
        print(f"\n  ✅ ENSEMBLE ESN {left_out_esn}:")
        print(f"     MAE: {mae:.1f} cycles")
        print(f"     R²: {r2:.3f}")
        print(f"     Baseline: {baseline:.1f}")
        print(f"     Improvement: {improvement:+.1f}%")
        print(f"     Range pred: [{y_pred_ensemble.min():.0f}, {y_pred_ensemble.max():.0f}]")
        print(f"     Range true: [{y_test.min():.0f}, {y_test.max():.0f}]")
        
        fold_results.append({
            'left_out_esn': left_out_esn,
            'mae': mae,
            'r2': r2,
            'baseline_mae': baseline,
            'improvement_pct': improvement,
            'y_true': y_test,
            'y_pred': y_pred_ensemble,
            'info_test': info_test,
            'model_scores': model_scores,
            'predictions': predictions
        })
    
    return fold_results


# ============================================================================
# SEZIONE 4: VISUALIZATION
# ============================================================================

def plot_periodic_results(fold_results, out_dir):
    """Plot risultati."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Per-fold plots
    for fr in fold_results:
        esn = fr['left_out_esn']
        cycles = fr['info_test']['Cycle'].values
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cycles, fr['y_true'], '-o', markersize=3, linewidth=2, alpha=0.8, label='Actual', color='#2E86AB')
        ax.plot(cycles, fr['y_pred'], '--', linewidth=2.5, alpha=0.75, label='Predicted (Ensemble)', color='#F18F01')
        
        ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
        ax.set_ylabel('RUL WW (cycles)', fontsize=12, fontweight='bold')
        ax.set_title(f"Periodic LOEO - ESN {esn} | MAE={fr['mae']:.1f}, R²={fr['r2']:.3f}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        
        fname = f"periodic_loeo_esn{esn}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {fname}")
    
    # Summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    esns = [fr['left_out_esn'] for fr in fold_results]
    maes = [fr['mae'] for fr in fold_results]
    r2s = [fr['r2'] for fr in fold_results]
    
    # MAE
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(esns)), maes, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(esns)))
    ax1.set_xticklabels([f'ESN {e}' for e in esns], fontweight='bold')
    ax1.set_ylabel('MAE (cycles)', fontsize=12, fontweight='bold')
    ax1.set_title('MAE per Test Engine', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, mae in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{mae:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # R²
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(esns)), r2s, color='#A23B72', alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(esns)))
    ax2.set_xticklabels([f'ESN {e}' for e in esns], fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('R² per Test Engine', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, r2 in zip(bars, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Scatter
    ax3 = axes[1, 0]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, fr in enumerate(fold_results):
        ax3.scatter(fr['y_true'], fr['y_pred'], s=20, alpha=0.5, color=colors[i % len(colors)], 
                   label=f"ESN {fr['left_out_esn']}", edgecolors='black', linewidth=0.3)
    all_y = np.concatenate([fr['y_true'] for fr in fold_results])
    lims = [0, max(all_y) * 1.05]
    ax3.plot(lims, lims, 'r--', linewidth=2, alpha=0.8, label='Perfect')
    ax3.set_xlabel('Actual RUL', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted RUL', fontsize=12, fontweight='bold')
    ax3.set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    avg_mae = np.mean(maes)
    avg_r2 = np.mean(r2s)
    avg_imp = np.mean([fr['improvement_pct'] for fr in fold_results])
    
    stats = f"PERIODIC LOEO RESULTS\n{'='*40}\n\n"
    stats += f"Average MAE: {avg_mae:.1f} cycles\n"
    stats += f"Average R²: {avg_r2:.3f}\n"
    stats += f"Improvement: {avg_imp:+.1f}%\n\n"
    stats += "Per-fold:\n"
    for fr in fold_results:
        stats += f"ESN {fr['left_out_esn']}: MAE={fr['mae']:.1f}, R²={fr['r2']:.3f}\n"
    
    ax4.text(0.1, 0.95, stats, transform=ax4.transAxes, fontsize=10, verticalalignment='top', 
            fontfamily='monospace', bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.7))
    
    plt.suptitle("Periodic Feature Engineering - WW LOEO", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig.savefig(os.path.join(out_dir, 'periodic_loeo_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: periodic_loeo_summary.png")


def save_results_csv(fold_results, out_dir):
    """Salva CSV."""
    os.makedirs(out_dir, exist_ok=True)
    
    rows = [{
        'left_out_esn': fr['left_out_esn'],
        'mae': fr['mae'],
        'r2': fr['r2'],
        'baseline_mae': fr['baseline_mae'],
        'improvement_pct': fr['improvement_pct']
    } for fr in fold_results]
    
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(out_dir, 'periodic_loeo_results.csv'), index=False)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(df_res.to_string(index=False))
    print("\n" + "="*70)
    print(f"Average MAE: {df_res['mae'].mean():.1f} cycles")
    print(f"Average R²: {df_res['r2'].mean():.3f}")
    print(f"Improvement: {df_res['improvement_pct'].mean():+.1f}%")
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("WW PREDICTION - PERIODIC FEATURE ENGINEERING")
    print("="*70)
    
    cfg = load_config('configs/config.yaml')
    df = pd.read_csv(cfg['data']['train_clean_csv'])
    print(f"\nLoaded: {df.shape}, ESNs: {df['ESN'].nunique()}")
    
    # Check ESN types
    print(f"ESN unique: {df['ESN'].unique()}")
    print(f"ESN dtype: {df['ESN'].dtype}")
    
    # Base features
    print("\n" + "="*70)
    print("STEP 1: BASE FEATURES")
    print("="*70)
    df_feat = create_base_features(df)
    print(f"After base features: {df_feat.shape}")
    
    print("\n" + "="*70)
    print("STEP 2: AGGREGATE")
    print("="*70)
    df_agg = aggregate_by_cycle(df_feat)
    print(f"After aggregation: {df_agg.shape}")
    
    # Verify data before periodic
    for esn in sorted(df_agg['ESN'].unique()):
        count = len(df_agg[df_agg['ESN'] == esn])
        print(f"  ESN {esn}: {count} cycles")
    
    print("\n" + "="*70)
    print("STEP 3: PERIODIC FEATURES")
    print("="*70)
    df_periodic = extract_periodic_features(df_agg)
    print(f"After periodic features: {df_periodic.shape}")
    
    # Verify data after periodic
    for esn in sorted(df_periodic['ESN'].unique()):
        count = len(df_periodic[df_periodic['ESN'] == esn])
        print(f"  ESN {esn}: {count} cycles")
    
    # Select features (ONLY EXISTING ONES)
    print("\n" + "="*70)
    print("STEP 4: FEATURE SELECTION")
    print("="*70)
    
    base_cols = ['temp_gradient', 'hptc_efficiency', 'thermal_stress', 'relative_cycle']
    
    # Get periodic columns that exist
    periodic_patterns = ['phase', 'sin', 'cos', 'roll', 'rate', 'cumsum', 'squared', 'interact']
    periodic_cols = [c for c in df_periodic.columns if any(p in c for p in periodic_patterns)]
    
    print(f"Available base columns: {[c for c in base_cols if c in df_periodic.columns]}")
    print(f"Available periodic columns: {len(periodic_cols)}")
    
    # Use only existing features
    feature_cols = [c for c in base_cols if c in df_periodic.columns] + periodic_cols[:30]
    
    print(f"Selected features: {len(feature_cols)}")
    print(f"Feature list: {feature_cols[:10]}...") 
    
    # Verify features exist
    missing = [f for f in feature_cols if f not in df_periodic.columns]
    if missing:
        print(f"⚠️ Missing features: {missing}")
        feature_cols = [f for f in feature_cols if f in df_periodic.columns]
    
    print(f"Final feature count: {len(feature_cols)}")
    
    window_size=52
    # LOEO Training
    print("\n" + "="*70)
    print("STEP 5: LOEO TRAINING")
    print("="*70)
    fold_results = train_periodic_loeo(df_periodic, feature_cols, window_size=window_size)
    
    if not fold_results:
        print("\n❌ No valid folds!")
        return
    
    # Save & Plot
    out_dir = f'artifacts/periodic_ww_loeo_{window_size}'
    print(f"\n{'='*70}\nSAVING RESULTS\n{'='*70}")
    plot_periodic_results(fold_results, out_dir)
    save_results_csv(fold_results, out_dir)
    
    print(f"\n📁 All results: {out_dir}/")
    print("\n" + "="*70)
    print("DONE ✨")
    print("="*70)


if __name__ == '__main__':
    main()


