import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\Afrano\Documents\Quera\ML project\preprocessing_output_v4.parquet"
df = pd.read_parquet(file_path)

rooms_map = {
    'بدون اتاق': 0, 'یک': 1, 'دو': 2, 'سه': 3, 'چهار': 4,
    'پنج': 5, 'شش': 6, 'هفت': 7, 'هشت': 8, 'نه': 9, 'ده': 10
}
if 'rooms_count' in df.columns:
    df['rooms_count'] = df['rooms_count'].replace(rooms_map)

selected = {
    'Price': 'price_value',
    'Land Size': 'land_size',
    'Building Size': 'building_size',
    'Rooms': 'rooms_count',
    'Latitude': 'location_latitude',
    'Longitude': 'location_longitude',
    'Building Age': 'building_age'
}

valid = {k: v for k, v in selected.items() if v in df.columns}
data = df[list(valid.values())].rename(columns={v: k for k, v in valid.items()})
data = data.apply(pd.to_numeric, errors='coerce')

for col in ['Price', 'Land Size', 'Building Size']:
    if col in data.columns:
        data.loc[data[col] <= 0, col] = np.nan

data = data.dropna(subset=['Price', 'Building Size'])

for col in ['Price', 'Land Size', 'Building Size', 'Building Age']:
    if col in data.columns:
        q_low = data[col].quantile(0.01)
        q_high = data[col].quantile(0.99)
        data = data[(data[col] >= q_low) & (data[col] <= q_high)]

corr_matrix = data.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Coefficient'}
)

plt.title("Correlation Matrix for Real Estate Features", fontsize=14, pad=15)
plt.tight_layout()
plt.show()