import pandas as pd
import numpy as np
import os
import sys

# Add the current directory to path so we can import utils
sys.path.append(os.getcwd())

from utils.data_loader import _generate_synthetic_data, load_data

print("--- Synthetic Data Check ---")
df_sync = _generate_synthetic_data()
print(f"Synthetic Total Rainfall: {df_sync['prcp'].sum():.2f}")

print("\n--- Real Data Check ---")
csv_path = "pakistan_weather_2000_2024.csv"
if os.path.exists(csv_path):
    df_real = pd.read_csv(csv_path)
    # The CSV uses 'prcp(Precipitation)' name before renaming
    col = [c for c in df_real.columns if 'prcp' in c.lower()][0]
    print(f"Real Total Rainfall ({col}): {df_real[col].sum():.2f}")
else:
    print(f"CSV not found at {csv_path}")

print("\n--- load_data() Check ---")
df_loaded = load_data()
# load_data renames it to 'prcp'
print(f"Loaded Data Total Rainfall: {df_loaded['prcp'].sum():.2f}")
if 'latitude' in df_loaded.columns:
    print(f"Sample Latitude: {df_loaded['latitude'].iloc[0]}")
