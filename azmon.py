import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\Afrano\Documents\Quera\ML project\preprocessing_output_v4.parquet"
df = pd.read_parquet(file_path)

# 1. پیدا کردن ستون سند تجاری و ستون قیمت
deed_col = None
for col in ['has_business_deed', 'deed_business_has', 'business_deed']:
    if col in df.columns:
        deed_col = col
        break

if deed_col is None:
    for col in df.columns:
        if 'business' in str(col).lower() and 'deed' in str(col).lower() and not str(col).endswith('_was_missing'):
            deed_col = col
            break

price_col = 'price_value' if 'price_value' in df.columns else 'price'

print(f"Selected deed column: {deed_col}")
print(f"Selected price column: {price_col}")

# 2. فیلتر و آماده‌سازی داده‌ها
analysis_df = df[[deed_col, price_col]].copy()
analysis_df[price_col] = pd.to_numeric(analysis_df[price_col], errors='coerce')
analysis_df[deed_col] = pd.to_numeric(analysis_df[deed_col], errors='coerce')

# حذف ردیف‌های خالی و قیمت‌های نامعتبر یا صفر
analysis_df = analysis_df.dropna()
analysis_df = analysis_df[analysis_df[price_col] > 0]

# 3. تفکیک گروه‌ها: دارای سند تجاری (1) و بدون سند تجاری (0)
group_with_deed = analysis_df[analysis_df[deed_col] == 1][price_col]
group_without_deed = analysis_df[analysis_df[deed_col] == 0][price_col]

count_with = len(group_with_deed)
count_without = len(group_without_deed)

mean_with = group_with_deed.mean()
mean_without = group_without_deed.mean()

median_with = group_with_deed.median()
median_without = group_without_deed.median()

# 4. اجرای آزمون فرض آماری T-test دو نمونه‌ای مستقل (Welch's t-test)
t_stat, p_value = stats.ttest_ind(group_with_deed, group_without_deed, equal_var=False)

# 5. چاپ نتایج و تفسیر آماری
print("\n" + "="*60)
print("                    نتایج آزمون فرض آماری")
print("="*60)
print(f"تعداد املاک با سند تجاری: {count_with:,}")
print(f"میانگین قیمت با سند تجاری: {mean_with:,.2f}")
print(f"میانه قیمت با سند تجاری: {median_with:,.2f}")
print("-" * 60)
print(f"تعداد املاک بدون سند تجاری: {count_without:,}")
print(f"میانگین قیمت بدون سند تجاری: {mean_without:,.2f}")
print(f"میانه قیمت بدون سند تجاری: {median_without:,.2f}")
print("-" * 60)
print(f"آماره تی (T-statistic): {t_stat:.4f}")
print(f"مقدار پی (P-value): {p_value:.4e}")
print("-" * 60)

alpha = 0.05
if p_value < alpha:
    print("نتیجه آماری: چون P-value کمتر از 0.05 است، فرض صفر (H0) رد می‌شود.")
    print("نتیجه‌گیری: داشتن سند تجاری تأثیر معناداری بر میانگین قیمت فروش ملک دارد.")
else:
    print("نتیجه آماری: چون P-value بیشتر از 0.05 است، فرض صفر (H0) رد نمی‌شود.")
    print("نتیجه‌گیری: تفاوت معناداری بین میانگین قیمت املاک دارای سند تجاری و فاقد آن مشاهده نشد.")
print("="*60)

# 6. رسم نمودار میله‌ای مقایسه میانگین قیمت
plt.figure(figsize=(7, 5))
sns.barplot(
    x=['Without Business Deed (0)', 'With Business Deed (1)'],
    y=[mean_without, mean_with],
    palette='Blues_d'
)
plt.title('Comparison of Average Price by Business Deed Status', fontsize=12, pad=12)
plt.ylabel('Average Price', fontsize=11)
plt.xlabel('Business Deed Status', fontsize=11)
plt.tight_layout()
plt.show()