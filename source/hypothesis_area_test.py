"""
Statistical Hypothesis Test: Residential Area (Metropolis vs. Small Cities/Villages)
H0: Mean residential area in metropolis >= Mean residential area in small cities
H1: Mean residential area in metropolis < Mean residential area in small cities
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path("data/processed/preprocessing_output_v4.parquet")
CITY_CLASS_PATH = Path("data/raw/iran_city_classification.csv")
OUTPUT_FIG_PATH = Path("outputs/figures/area_metropolis_vs_small.png")
ALPHA = 0.01


def load_and_prepare_data(parquet_path: Path, city_csv_path: Path) -> pd.DataFrame:
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # 1. Filter to Residential only (Apartment & Villa)
    residential_cols = [
        c
        for c in ["property_category_apartment", "property_category_villa"]
        if c in df.columns
    ]
    if residential_cols:
        is_residential = df[residential_cols].sum(axis=1) > 0
        df = df[is_residential].copy()
        print(f"Filtered to residential properties: {len(df):,} records remaining.")

    # 2. Extract Area Column
    area_col = None
    for candidate in [
        "building_size",
        "area",
        "meterage",
        "size",
        "clean_building_size",
    ]:
        if candidate in df.columns:
            area_col = candidate
            break

    if not area_col:
        raise ValueError(f"No suitable area column found among: {df.columns.tolist()}")

    df["area_clean"] = pd.to_numeric(df[area_col], errors="coerce")
    # Trimming realistic residential area (25m² to 800m²)
    df = df.dropna(subset=["area_clean"])
    df = df[(df["area_clean"] >= 25) & (df["area_clean"] <= 800)].copy()

    # 3. Read City Classification
    df_city = pd.read_csv(city_csv_path)
    city_col = df_city.columns[0]
    cat_col = df_city.columns[1]
    df_city[city_col] = df_city[city_col].astype(str).str.strip().str.lower()

    # 4. Merge City Data
    divar_city_col = "city_slug" if "city_slug" in df.columns else "city"
    df["city_clean"] = (
        df[divar_city_col].astype(str).str.strip().str.lower().str.replace(" ", "-")
    )

    merged = df.merge(
        df_city,
        left_on="city_clean",
        right_on=city_col,
        how="inner",
    )

    merged["is_metropolis"] = merged[cat_col].astype(str).str.strip() == "کلان‌شهر"
    merged["group_label"] = np.where(
        merged["is_metropolis"],
        "Metropolis (کلان‌شهر)",
        "Small City/Village (شهر کوچک)",
    )

    return merged


def run_hypothesis_test(df: pd.DataFrame):
    metro_area = df[df["is_metropolis"]]["area_clean"].values
    small_area = df[~df["is_metropolis"]]["area_clean"].values

    n_metro, n_small = len(metro_area), len(small_area)
    mean_m, mean_s = np.mean(metro_area), np.mean(small_area)
    std_m, std_s = np.std(metro_area, ddof=1), np.std(small_area, ddof=1)
    med_m, med_s = np.median(metro_area), np.median(small_area)

    print("\n" + "=" * 60)
    print(" " * 18 + "DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(
        f"Metropolis:    N = {n_metro:,} | Mean = {mean_m:.2f} m² | Median = {med_m:.1f} m² | Std = {std_m:.2f}"
    )
    print(
        f"Small Cities:  N = {n_small:,} | Mean = {mean_s:.2f} m² | Median = {med_s:.1f} m² | Std = {std_s:.2f}"
    )
    print(f"Difference in Means (Metro - Small): {mean_m - mean_s:.2f} m²")

    # One-sided Welch's t-test (alternative='less' -> H1: mean(metro) < mean(small))
    t_stat, p_val_t = stats.ttest_ind(
        metro_area, small_area, equal_var=False, alternative="less"
    )

    # Mann-Whitney U test (Non-parametric check)
    u_stat, p_val_u = stats.mannwhitneyu(metro_area, small_area, alternative="less")

    # Effect Size: Cohen's d
    pooled_sd = np.sqrt(
        ((n_metro - 1) * std_m**2 + (n_small - 1) * std_s**2) / (n_metro + n_small - 2)
    )
    cohens_d = (mean_m - mean_s) / pooled_sd

    print("\n" + "=" * 60)
    print(" " * 20 + "HYPOTHESIS TEST RESULTS")
    print("=" * 60)
    print(f"Welch's t-statistic : {t_stat:.4f}")
    print(f"p-value (Welch's)   : {p_val_t:.4e}")
    print(f"Mann-Whitney U p-val: {p_val_u:.4e}")
    print(f"Cohen's d           : {cohens_d:.4f}")
    print("-" * 60)

    if p_val_t < ALPHA and mean_m < mean_s:
        print("✅ RESULT: The null hypothesis is REJECTED (p < 0.01).")
        print("   -> The dataset SUPPORTS the hypothesis that residential units")
        print("      in metropolises have significantly smaller mean area.")
    else:
        print("❌ RESULT: Fail to reject the null hypothesis.")
        print(
            "   -> The dataset DOES NOT support the claim of smaller area in metropolises."
        )
    print("=" * 60 + "\n")

    return metro_area, small_area


def plot_results(metro_area: np.ndarray, small_area: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Density Histogram / KDE
    axes[0].hist(
        metro_area,
        bins=50,
        range=(25, 300),
        density=True,
        alpha=0.55,
        color="#2b5c8f",
        label=f"Metropolis (Mean: {np.mean(metro_area):.1f}m²)",
    )
    axes[0].hist(
        small_area,
        bins=50,
        range=(25, 300),
        density=True,
        alpha=0.55,
        color="#d95f02",
        label=f"Small Cities (Mean: {np.mean(small_area):.1f}m²)",
    )
    axes[0].set_title("Distribution of Residential Area", fontsize=12)
    axes[0].set_xlabel("Area (m²)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Boxplot
    axes[1].boxplot(
        [metro_area, small_area],
        labels=["Metropolis", "Small Cities/Villages"],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#ccebc5", color="#2b5c8f"),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1].set_title("Area Comparison (Boxplot - Outliers Hidden)", fontsize=12)
    axes[1].set_ylabel("Area (m²)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    df_prepared = load_and_prepare_data(DATA_PATH, CITY_CLASS_PATH)
    metro_arr, small_arr = run_hypothesis_test(df_prepared)
    plot_results(metro_arr, small_arr, OUTPUT_FIG_PATH)
