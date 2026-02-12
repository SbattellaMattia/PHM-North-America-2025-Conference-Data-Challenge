import pandas as pd
import matplotlib.pyplot as plt

# Load the targeted cleaned data
df = pd.read_csv('/Users/niccolociotti/Desktop/PHM-North-America-2025-Conference-Data-Challenge/data/outlier_cleaned_data.csv')

# Identify sensor columns
sensor_cols = [col for col in df.columns if col.startswith('Sensed_')]

# Initialize a dataframe for the standardized data
df_standardized = df.copy()

# --- FUNZIONE DI STANDARDIZZAZIONE ---
def standardize_sensors_by_snapshot(df):
    df_std = df.copy()
    # Seleziona solo i sensori
    sensor_cols = [col for col in df.columns if col.startswith('Sensed_')]
    
    for col in sensor_cols:
        # Calcola statistiche per gruppo
        means = df.groupby('Snapshot')[col].transform('mean')
        stds = df.groupby('Snapshot')[col].transform('std')
        
        # Evita divisioni per zero
        stds = stds.replace(0, 1)
        
        # Applica Z-Score
        df_std[col] = (df[col] - means) / stds
        
    return df_std

# Applica Standardizzazione
df_final = standardize_sensors_by_snapshot(df)

# 3. Salva il CSV finale
df_final.to_csv('/Users/niccolociotti/Desktop/PHM-North-America-2025-Conference-Data-Challenge/data/standardized_cleaned_data.csv', index=False)

# Summary statistics for the user
stats = df_standardized[sensor_cols].agg(['mean', 'std']).round(4)
print("Standardization Summary (First 5 sensors):")
print(stats.iloc[:, :5])

# Visualization: Boxplot to show that all sensors are now on the same scale
plt.figure(figsize=(14, 6))
df_standardized[sensor_cols].boxplot()
plt.xticks(rotation=45, ha='right')
plt.title('Distribuzione dei Sensori dopo la Standardizzazione (Z-score)')
plt.ylabel('Valore Standardizzato')
plt.tight_layout()
plt.show()