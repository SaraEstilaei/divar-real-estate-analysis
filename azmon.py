import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\Afrano\Documents\Quera\ML project\preprocessing_output_v3.parquet"
df = pd.read_parquet(file_path)

data = df[['price_value', 'has_business_deed']].dropna().copy()
data['price_value'] = pd.to_numeric(data['price_value'], errors='coerce')
data = data[data['price_value'] > 0]

print("Values in has_business_deed column:")
print(data['has_business_deed'].value_counts(dropna=False))

data['has_business_deed_clean'] = data['has_business_deed'].astype(str).str.lower().map({
    'true': 1, '1': 1, '1.0': 1,
    'false': 0, '0': 0, '0.0': 0
})

data = data.dropna(subset=['has_business_deed_clean'])

q_low = data['price_value'].quantile(0.01)
q_high = data['price_value'].quantile(0.99)
data_filtered = data[(data['price_value'] >= q_low) & (data['price_value'] <= q_high)]

with_deed = data_filtered[data_filtered['has_business_deed_clean'] == 1]['price_value']
without_deed = data_filtered[data_filtered['has_business_deed_clean'] == 0]['price_value']

print("-" * 40)
print("With Business Deed - Count:", len(with_deed))
print("With Business Deed - Mean Price:", with_deed.mean())
print("Without Business Deed - Count:", len(without_deed))
print("Without Business Deed - Mean Price:", without_deed.mean())

t_stat, p_value = stats.ttest_ind(with_deed, without_deed, equal_var=False)

print("-" * 40)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4e}")

if p_value < 0.05:
    print("Result: Statistically Significant (p < 0.05). Having a business deed significantly affects the sale price.")
else:
    print("Result: Not Statistically Significant (p >= 0.05).")

plt.figure(figsize=(8, 5))
sns.barplot(x='has_business_deed_clean', y='price_value', data=data_filtered, hue='has_business_deed_clean', errorbar=None, palette='Blues', legend=False)
plt.title('Average Sale Price by Business Deed Status')
plt.xlabel('Has Business Deed')
plt.ylabel('Average Price')
plt.xticks([0, 1], ['Without Deed', 'With Deed'])
plt.show()