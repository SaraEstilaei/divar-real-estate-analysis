import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(current_dir, 'Divar.csv')
output_csv = os.path.join(current_dir, 'preprocessed_divar.csv')

print("1. Reading data...")
df = pd.read_csv(input_csv)

cols = ['land_size', 'building_size', 'floor', 'rooms_count', 
        'total_floors_count', 'unit_per_floor', 'construction_year', 
        'is_rebuilt', 'deed_type', 'has_business_deed']
df_copy = df[cols].copy()

print("2. Converting Persian numbers and preprocessing...")
year_replacements = {'قبل از ۱۳۷۰': '1365', 'قبل از 1370': '1365'}
df_copy['construction_year'] = df_copy['construction_year'].replace(year_replacements)


persian_to_english = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
df_copy['construction_year'] = df_copy['construction_year'].astype(str).str.translate(persian_to_english)


df_copy['construction_year'] = pd.to_numeric(df_copy['construction_year'], errors='coerce')

print("3. Saving CSV...")
df_copy.to_csv(output_csv, index=False, encoding='utf-8-sig')

print("4. Plotting histogram...")

valid_years = df_copy['construction_year'].dropna()
valid_years = valid_years[(valid_years >= 1365) & (valid_years <= 1404)]

plt.figure(figsize=(12, 6))

bins = range(1365, 1406)
plt.hist(valid_years, bins=bins, color='skyblue', edgecolor='black', align='left')

plt.title('Construction Year Histogram (1365 - 1404)')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(range(1365, 1405, 5))
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()