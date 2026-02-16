import numpy as np
import pandas as pd
import warnings
from itertools import combinations, permutations
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import os
from src.utils import load_config

warnings.filterwarnings('ignore')

def generate_exhaustive_features(df, snapshot=4):
    """
    Genera feature ULTRA-EXHAUSTIVE con combinazioni avanzate.
    """
    d = df[df['Snapshot'] == snapshot].copy()
    sensor_cols = [c for c in d.columns if c.startswith('Sensed_')]
    
    print(f"Base sensors: {len(sensor_cols)}")
    
    features = {}
    
    # ============================================================================
    # 1) PAIRWISE OPERATIONS (esteso: 8 operazioni invece di 4)
    # ============================================================================
    print("\nGenerating pairwise features...")
    count = 0
    for s1, s2 in combinations(sensor_cols, 2):
        denom_s2 = d[s2].abs() + 1e-6
        denom_s1 = d[s1].abs() + 1e-6
        
        # Base operations
        features[f'ratio_{s1}_{s2}'] = d[s1] / denom_s2
        features[f'ratio_inv_{s1}_{s2}'] = d[s2] / denom_s1
        features[f'diff_{s1}_{s2}'] = d[s1] - d[s2]
        features[f'sum_{s1}_{s2}'] = d[s1] + d[s2]
        features[f'product_{s1}_{s2}'] = d[s1] * d[s2]
        
        # Advanced operations
        features[f'harmonic_mean_{s1}_{s2}'] = 2 / (1/denom_s1 + 1/denom_s2)
        features[f'geometric_mean_{s1}_{s2}'] = np.sqrt(d[s1].abs() * d[s2].abs() + 1e-6)
        features[f'ratio_squared_{s1}_{s2}'] = (d[s1] / denom_s2) ** 2
        
        count += 8
    
    print(f"  Pairwise: {count} features")
    
    # ============================================================================
    # 2) UNIVARIATE TRANSFORMS (esteso: 10 operazioni)
    # ============================================================================
    print("Generating univariate transforms...")
    count = 0
    for s in sensor_cols:
        shift = max(0, 1 - d[s].min())
        
        # Standard transforms
        features[f'log_{s}'] = np.log(d[s] + shift + 1e-6)
        features[f'sqrt_{s}'] = np.sqrt(d[s] + shift)
        features[f'square_{s}'] = d[s] ** 2
        features[f'cube_{s}'] = d[s] ** 3
        features[f'inv_{s}'] = 1 / (d[s].abs() + 1e-6)
        
        # Advanced transforms
        features[f'log10_{s}'] = np.log10(d[s] + shift + 1e-6)
        features[f'exp_{s}'] = np.exp((d[s] - d[s].mean()) / (d[s].std() + 1e-6))  # Standardized exp
        features[f'power4_{s}'] = d[s] ** 4
        features[f'cbrt_{s}'] = np.sign(d[s]) * np.abs(d[s]) ** (1/3)  # Cube root
        features[f'reciprocal_sqrt_{s}'] = 1 / (np.sqrt(d[s].abs() + 1e-6) + 1e-6)
        
        count += 10
    
    print(f"  Univariate: {count} features")
    
    # ============================================================================
    # 3) TRIPLET COMBINATIONS (esteso: 9 operazioni invece di 3)
    # ============================================================================
    print("Generating triplet features...")
    count = 0
    np.random.seed(42)
    sampled = np.random.choice(len(sensor_cols), size=min(10, len(sensor_cols)), replace=False)
    
    for i in sampled:
        for j in sampled:
            for k in sampled:
                if i < j < k:
                    s1, s2, s3 = sensor_cols[i], sensor_cols[j], sensor_cols[k]
                    
                    # Standard triplets
                    features[f'tri_sum_ratio_{s1}_{s2}_{s3}'] = (d[s1] + d[s2]) / (d[s3].abs() + 1e-6)
                    features[f'tri_diff_ratio_{s1}_{s2}_{s3}'] = (d[s1] - d[s2]) / (d[s3].abs() + 1e-6)
                    features[f'tri_prod_ratio_{s1}_{s2}_{s3}'] = (d[s1] * d[s2]) / (d[s3].abs() + 1e-6)
                    
                    # Advanced triplets
                    features[f'tri_sum_prod_{s1}_{s2}_{s3}'] = (d[s1] + d[s2]) * d[s3]
                    features[f'tri_ratio_sum_{s1}_{s2}_{s3}'] = d[s1] / ((d[s2] + d[s3]).abs() + 1e-6)
                    features[f'tri_ratio_diff_{s1}_{s2}_{s3}'] = d[s1] / ((d[s2] - d[s3]).abs() + 1e-6)
                    features[f'tri_mean_ratio_{s1}_{s2}_{s3}'] = (d[s1] + d[s2]) / 2 / (d[s3].abs() + 1e-6)
                    features[f'tri_weighted_{s1}_{s2}_{s3}'] = (d[s1] * 2 + d[s2]) / (d[s3].abs() + 1e-6)
                    features[f'tri_harmonic_{s1}_{s2}_{s3}'] = 3 / (1/(d[s1].abs()+1e-6) + 1/(d[s2].abs()+1e-6) + 1/(d[s3].abs()+1e-6))
                    
                    count += 9
    
    print(f"  Triplets: {count} features")
    
    # ============================================================================
    # 4) QUADRUPLET COMBINATIONS (NUOVO)
    # ============================================================================
    print("Generating quadruplet features...")
    count = 0
    sampled_quad = np.random.choice(len(sensor_cols), size=min(6, len(sensor_cols)), replace=False)
    
    for i in sampled_quad:
        for j in sampled_quad:
            for k in sampled_quad:
                for l in sampled_quad:
                    if i < j < k < l:
                        s1, s2, s3, s4 = sensor_cols[i], sensor_cols[j], sensor_cols[k], sensor_cols[l]
                        
                        features[f'quad_ratio_{s1}_{s2}_{s3}_{s4}'] = (d[s1] * d[s2]) / ((d[s3] * d[s4]).abs() + 1e-6)
                        features[f'quad_sum_ratio_{s1}_{s2}_{s3}_{s4}'] = (d[s1] + d[s2]) / ((d[s3] + d[s4]).abs() + 1e-6)
                        features[f'quad_diff_ratio_{s1}_{s2}_{s3}_{s4}'] = (d[s1] - d[s2]) / ((d[s3] - d[s4]).abs() + 1e-6)
                        
                        count += 3
    
    print(f"  Quadruplets: {count} features")
    
    # ============================================================================
    # 5) TEMPERATURE/PRESSURE PHYSICS (esteso)
    # ============================================================================
    temp_sensors = [s for s in sensor_cols if 'T' in s and 'TAT' not in s]
    press_sensors = [s for s in sensor_cols if 'P' in s]
    
    print("Generating physics features...")
    count = 0
    
    # Temperature features
    for t1, t2 in combinations(temp_sensors, 2):
        features[f'temp_increment_{t1}_{t2}'] = (d[t2] - d[t1]) / (d[t1] + 1e-6)
        features[f'temp_ratio_{t1}_{t2}'] = d[t1] / (d[t2] + 1e-6)
        features[f'temp_diff_{t1}_{t2}'] = d[t1] - d[t2]
        features[f'temp_sum_{t1}_{t2}'] = d[t1] + d[t2]
        features[f'temp_product_{t1}_{t2}'] = d[t1] * d[t2]
        features[f'temp_increment_squared_{t1}_{t2}'] = ((d[t2] - d[t1]) / (d[t1] + 1e-6)) ** 2
        count += 6
    
    # Pressure features
    for p1, p2 in combinations(press_sensors, 2):
        features[f'press_ratio_{p1}_{p2}'] = d[p1] / (d[p2] + 1e-6)
        features[f'press_diff_{p1}_{p2}'] = d[p1] - d[p2]
        features[f'press_sum_{p1}_{p2}'] = d[p1] + d[p2]
        features[f'press_product_{p1}_{p2}'] = d[p1] * d[p2]
        count += 4
    
    # Temperature-Pressure cross
    for t in temp_sensors[:3]:  # Limit to avoid explosion
        for p in press_sensors[:3]:
            features[f'temp_press_ratio_{t}_{p}'] = d[t] / (d[p] + 1e-6)
            features[f'temp_press_product_{t}_{p}'] = d[t] * d[p]
            count += 2
    
    # Isentropic efficiency variants
    if all(s in sensor_cols for s in ['Sensed_T25', 'Sensed_T3', 'Sensed_Ps3', 'Sensed_Pt2']):
        gamma = 1.4
        try:
            # Standard isentropic
            features['isentropic_eff'] = (
                d['Sensed_T25'] * (np.power(d['Sensed_Ps3'] / (d['Sensed_Pt2'] + 1e-6), (gamma-1)/gamma) - 1)
            ) / (d['Sensed_T3'] - d['Sensed_T25'] + 1e-6)
            
            # Squared variant
            features['isentropic_eff_squared'] = features['isentropic_eff'] ** 2
            
            # Inverse
            features['isentropic_eff_inv'] = 1 / (features['isentropic_eff'].abs() + 1e-6)
            
            count += 3
        except:
            pass
    
    print(f"  Physics: {count} features")
    
    # ============================================================================
    # 6) OPERATIONAL CONTEXT (esteso: normalization + interactions)
    # ============================================================================
    if 'Sensed_Altitude' in sensor_cols and 'Sensed_Mach' in sensor_cols:
        print("Generating operational features...")
        count = 0
        
        for s in sensor_cols:
            if s not in ['Sensed_Altitude', 'Sensed_Mach']:
                # Normalization
                features[f'norm_altitude_{s}'] = d[s] / (d['Sensed_Altitude'].abs() + 1e-6)
                features[f'norm_mach_{s}'] = d[s] / (d['Sensed_Mach'].abs() + 1e-6)
                
                # Interactions
                features[f'interact_alt_{s}'] = d[s] * d['Sensed_Altitude']
                features[f'interact_mach_{s}'] = d[s] * d['Sensed_Mach']
                features[f'interact_alt_mach_{s}'] = d[s] * d['Sensed_Altitude'] * d['Sensed_Mach']
                
                # Squared interactions
                features[f'interact_alt_squared_{s}'] = d[s] * (d['Sensed_Altitude'] ** 2)
                features[f'interact_mach_squared_{s}'] = d[s] * (d['Sensed_Mach'] ** 2)
                
                count += 7
        
        print(f"  Operational: {count} features")
    
    # ============================================================================
    # 7) POLYNOMIAL FEATURES (selected pairs, degree 2)
    # ============================================================================
    print("Generating polynomial features (degree 2)...")
    count = 0
    important_sensors = [s for s in sensor_cols if any(x in s for x in ['T3', 'T45', 'T5', 'Ps3', 'P25'])]
    
    for s1, s2 in combinations(important_sensors[:5], 2):  # Limit to most important
        denom = (d[s1].abs() + d[s2].abs() + 1e-6)
        features[f'poly_{s1}_{s2}'] = (d[s1]**2 + d[s2]**2) / denom
        features[f'poly_cross_{s1}_{s2}'] = (d[s1] * d[s2]) / denom
        count += 2
    
    print(f"  Polynomial: {count} features")
    
    # ============================================================================
    # 8) RATIOS OF RATIOS (meta-features)
    # ============================================================================
    print("Generating meta-features (ratios of ratios)...")
    count = 0
    
    # Pre-compute important ratios
    ratio_T3_T45 = d['Sensed_T3'] / (d['Sensed_T45'] + 1e-6)
    ratio_T3_T5 = d['Sensed_T3'] / (d['Sensed_T5'] + 1e-6)
    ratio_Ps3_P25 = d['Sensed_Ps3'] / (d['Sensed_P25'] + 1e-6)
    
    features['meta_ratio_T3T45_T3T5'] = ratio_T3_T45 / (ratio_T3_T5 + 1e-6)
    features['meta_ratio_T3T45_Ps3P25'] = ratio_T3_T45 / (ratio_Ps3_P25 + 1e-6)
    features['meta_ratio_T3T5_Ps3P25'] = ratio_T3_T5 / (ratio_Ps3_P25 + 1e-6)
    features['meta_product_T3T45_Ps3P25'] = ratio_T3_T45 * ratio_Ps3_P25
    features['meta_diff_T3T45_T3T5'] = ratio_T3_T45 - ratio_T3_T5
    
    count += 5
    print(f"  Meta-features: {count} features")
    
    # Convert to DataFrame
    df_features = pd.DataFrame(features, index=d.index)
    
    # Combine with ESN, Cycles, targets
    result = pd.concat([
        d[['ESN', 'Cycles']],
        df_features,
        d[['Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']]
    ], axis=1)
    
    return result


def plot_best_features_scatter(df_clean, df_corr_matrix, targets, out_dir):
    """
    Genera scatter plot per la miglior feature di ogni target.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    
    print("\n" + "="*70)
    print("GENERATING SCATTER PLOTS FOR BEST FEATURES")
    print("="*70)
    
    for target in targets:
        # Trova best feature per questo target
        best_idx = df_corr_matrix[target].abs().idxmax()
        best_feat = df_corr_matrix.loc[best_idx, 'feature']
        best_corr = df_corr_matrix.loc[best_idx, target]
        
        print(f"\n{target}:")
        print(f"  Best feature: {best_feat}")
        print(f"  Correlation: {best_corr:+.4f}")
        
        # Check if feature exists in df_clean
        if best_feat not in df_clean.columns:
            print(f"  ⚠️ Feature not found in cleaned data, skipping...")
            continue
        
        # Create figure con 2x2 subplot (uno per ESN)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        esns = sorted(df_clean['ESN'].unique())
        
        for i, esn in enumerate(esns):
            ax = axes[i]
            d_esn = df_clean[df_clean['ESN'] == esn].copy()
            
            # Remove NaN/Inf
            mask = ~(d_esn[best_feat].isna() | np.isinf(d_esn[best_feat]) | 
                     d_esn[target].isna() | np.isinf(d_esn[target]))
            d_esn = d_esn[mask]
            
            if len(d_esn) < 10:
                ax.text(0.5, 0.5, f'ESN {esn}\nInsufficient data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                continue
            
            # Scatter con colormap per Cycles
            scatter = ax.scatter(
                d_esn[best_feat],
                d_esn[target],
                c=d_esn['Cycles'],
                cmap='viridis',
                s=20,
                alpha=0.7,
                edgecolors='none'
            )
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cycles', fontsize=9)
            
            # Linear fit
            corr_esn, pval = pearsonr(d_esn[best_feat], d_esn[target])
            z = np.polyfit(d_esn[best_feat], d_esn[target], 1)
            p = np.poly1d(z)
            
            x_line = np.linspace(d_esn[best_feat].min(), d_esn[best_feat].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2.5, alpha=0.9,
                   label=f'Corr={corr_esn:+.3f} (p={pval:.2e})')
            
            # R²
            y_pred = p(d_esn[best_feat])
            r2 = r2_score(d_esn[target], y_pred)
            
            # Text box con stats
            textstr = f'Pearson: {corr_esn:+.3f}\nR²: {r2:.3f}\nN: {len(d_esn)}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            ax.set_xlabel(best_feat[:50], fontsize=10, fontweight='bold')
            ax.set_ylabel(target, fontsize=10, fontweight='bold')
            ax.set_title(f'ESN {esn}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)
        
        # Suptitle con info globale
        fig.suptitle(
            f'Best Feature for {target}\n'
            f'Feature: {best_feat}\n'
            f'Global Correlation: {best_corr:+.4f}',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save
        safe_target = target.replace('_', '').replace('Cycles', '').replace('to', '')
        safe_feat = best_feat.replace('/', '_').replace(' ', '_')[:50]
        
        fname = f"scatter_best_{safe_target}_{safe_feat}.png"
        fig.savefig(f"{out_dir}/{fname}", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✅ Saved: {fname}")
    
    print("\n" + "="*70)


def main():
    
    # Load data
    cfg = load_config("configs/config.yaml")
    df = pd.read_csv(cfg["data"]["train_csv"])
    print(f"\nLoaded: {df.shape[0]} rows, {df['ESN'].nunique()} ESNs")
    
    # Generate features
    print("\n" + "="*70)
    print("STEP 1: FEATURE GENERATION (Snapshot 4)")
    print("="*70)
    
    df_features = generate_exhaustive_features(df, snapshot=4)
    
    # Aggregate by cycle
    print("\nAggregating by cycle...")
    feature_cols = [c for c in df_features.columns 
                    if c not in ['ESN', 'Cycles', 'Cycles_to_WW', 
                                 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']]
    
    agg_dict = {f: 'mean' for f in feature_cols}
    agg_dict.update({
        'Cycles_to_WW': 'first',
        'Cycles_to_HPC_SV': 'first',
        'Cycles_to_HPT_SV': 'first'
    })
    
    df_agg = df_features.groupby(['ESN', 'Cycles']).agg(agg_dict).reset_index()
    
    print(f"Features generated: {len(feature_cols)}")
    print(f"Samples after aggregation: {len(df_agg)}")
    
    # Clean NaN/Inf
    print("\nCleaning NaN/Inf...")
    df_clean = df_agg.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Samples after cleaning: {len(df_clean)}")
    
    # ============================================================================
    # STEP 2: COMPUTE CORRELATION MATRIX (Pearson only, for simplicity)
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 2: COMPUTING CORRELATION MATRIX")
    print("="*70)
    
    targets = ['Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']
    
    # Build matrix: features × targets
    corr_data = []
    
    for feat in feature_cols:
        row = {'feature': feat}
        
        for target in targets:
            # Get clean data
            mask = ~(df_clean[feat].isna() | np.isinf(df_clean[feat]) | 
                df_clean[target].isna() | np.isinf(df_clean[target]))
            
            if mask.sum() > 10:
                try:
                    corr, _ = pearsonr(df_clean[feat][mask], df_clean[target][mask])
                    row[target] = corr
                except:
                    row[target] = np.nan
            else:
                row[target] = np.nan
        
        corr_data.append(row)
    
    df_corr_matrix = pd.DataFrame(corr_data)
    
    # Add absolute max correlation column
    df_corr_matrix['abs_max_corr'] = df_corr_matrix[targets].abs().max(axis=1)
    
    # Sort by abs_max_corr
    df_corr_matrix = df_corr_matrix.sort_values('abs_max_corr', ascending=False)
    
    # Output directory
    out_dir = "artifacts/exhaustive_features"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save full matrix
    df_corr_matrix.to_csv(f"{out_dir}/correlation_matrix_full.csv", index=False)
    print(f"\nFull correlation matrix saved: {out_dir}/correlation_matrix_full.csv")
    print(f"  Shape: {df_corr_matrix.shape}")
    
    # Print top-50
    print("\n" + "="*70)
    print("TOP-50 FEATURES BY MAX ABSOLUTE CORRELATION")
    print("="*70)
    print(f"\n{'Rank':<5} {'Feature':<65} {'WW':<10} {'HPC':<10} {'HPT':<10} {'Max':<10}")
    print("="*110)
    
    for idx, row in df_corr_matrix.head(50).iterrows():
        print(f"{idx+1:<5} {row['feature'][:64]:<65} "
              f"{row['Cycles_to_WW']:>9.4f} "
              f"{row['Cycles_to_HPC_SV']:>9.4f} "
              f"{row['Cycles_to_HPT_SV']:>9.4f} "
              f"{row['abs_max_corr']:>9.4f}")
    
    # Save top-100
    df_corr_matrix.head(100).to_csv(f"{out_dir}/correlation_matrix_top100.csv", index=False)
    print(f"\nTop-100 saved: {out_dir}/correlation_matrix_top100.csv")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for target in targets:
        max_corr = df_corr_matrix[target].abs().max()
        best_feat = df_corr_matrix.loc[df_corr_matrix[target].abs().idxmax(), 'feature']
        best_val = df_corr_matrix.loc[df_corr_matrix[target].abs().idxmax(), target]
        
        print(f"\n{target}:")
        print(f"  Max |correlation|: {max_corr:.4f}")
        print(f"  Best feature: {best_feat}")
        print(f"  Correlation value: {best_val:+.4f}")
    
    # ============================================================================
    # STEP 3: GENERATE SCATTER PLOTS (NUOVO)
    # ============================================================================
    plot_best_features_scatter(df_clean, df_corr_matrix, targets, out_dir)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"\n📁 All results saved in: {out_dir}/")
    print("  - correlation_matrix_full.csv")
    print("  - correlation_matrix_top100.csv")
    print("  - scatter_best_WW_*.png")
    print("  - scatter_best_HPCSV_*.png")
    print("  - scatter_best_HPTSV_*.png")


if __name__ == "__main__":
    main()
