import pandas as pd
import os

path = 'data/regimes/regime_labels.csv'
print(f"Checking file at: {os.path.abspath(path)}")

try:
    df = pd.read_csv(path)
    print(f"Start: {df['Date'].min()}")
    print(f"End: {df['Date'].max()}")
    print(f"Shape: {df.shape}")
except Exception as e:
    print(f"Error: {e}")

