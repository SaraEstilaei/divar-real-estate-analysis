"""
Price Analysis Module:
1. Inflation adjustment (Nominal vs. Real prices)
2. Price distribution across Level-3 categories
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Central Bank / Statistical Center CPI (Base 1400 = 100)
DEFAULT_CPI: Dict[int, float] = {
    1400: 100.0,
    1401: 145.8,
    1402: 205.1,
    1403: 270.7,
}

CATEGORY_COLS: List[str] = [
    "property_category_apartment",
    "property_category_villa",
    "property_category_office",
    "property_category_shop",
    "property_category_plot_land",
    "property_category_industry",
    "property_category_other",
]


def decode_categories(df: pd.DataFrame) -> pd.Series:
    """Decodes one-hot encoded category columns into a single string label."""
    if "category_level_3" in df.columns:
        return df["category_level_3"].astype(str)

    available_cols = [c for c in CATEGORY_COLS if c in df.columns]
    if not available_cols:
        return pd.Series("unknown", index=df.index)

    cat_matrix = df[available_cols].fillna(0)
    max_col = cat_matrix.idxmax(axis=1)
    has_category = cat_matrix.max(axis=1) > 0

    clean_names = {c: c.replace("property_category_", "") for c in available_cols}
    decoded = max_col.map(clean_names)
    decoded[~has_category] = "unknown"
    return decoded


def extract_year_column(df: pd.DataFrame) -> pd.Series:
    """Extracts ad/deal year; falls back to construction year if ad date is missing."""
    # Priority: Date of listing / ad creation
    for col in ["created_year", "ad_year", "year", "deal_year"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    # If datetime column exists
    for col in ["created_at", "date", "publish_date"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce").dt.year

    # Fallback to construction_year
    if "construction_year" in df.columns:
        # Convert Persian/Arabic digits
        persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        trans = str.maketrans(persian_digits + arabic_digits, "0123456789" * 2)
        return (
            df["construction_year"]
            .astype(str)
            .str.translate(trans)
            .str.extract(r"(\d{4})")[0]
            .astype(float)
        )

    raise ValueError("No valid date/year column found in dataframe.")


def prepare_sales_data(
    df: pd.DataFrame,
    min_year: int = 1400,
    max_year: int = 1403,
) -> pd.DataFrame:
    """Filters sales ads with positive price within valid years."""
    # Filter sales
    if "deal_type_sell" in df.columns:
        df_sales = df[df["deal_type_sell"] == 1].copy()
    else:
        df_sales = df.copy()

    df_sales = df_sales[
        pd.to_numeric(df_sales["price_value"], errors="coerce") > 0
    ].copy()
    df_sales["clean_year"] = extract_year_column(df_sales)
    df_sales = df_sales[df_sales["clean_year"].between(min_year, max_year)].copy()
    df_sales["clean_year"] = df_sales["clean_year"].astype(int)
    df_sales["property_type"] = decode_categories(df_sales)

    return df_sales


def compute_real_prices(
    df: pd.DataFrame,
    cpi_map: Dict[int, float] = DEFAULT_CPI,
    base_year: int = 1400,
) -> pd.DataFrame:
    """
    Computes nominal vs. real prices:
    Real Price = Nominal Price * (CPI_base / CPI_current)
    """
    summary = (
        df.groupby("clean_year")["price_value"]
        .agg(
            count="count",
            mean_nominal="mean",
            median_nominal="median",
        )
        .reset_index()
    )

    summary = summary[summary["clean_year"].isin(cpi_map.keys())].copy()
    summary["cpi"] = summary["clean_year"].map(cpi_map)

    base_cpi = cpi_map[base_year]
    summary["mean_real"] = summary["mean_nominal"] * (base_cpi / summary["cpi"])
    summary["median_real"] = summary["median_nominal"] * (base_cpi / summary["cpi"])

    return summary


def plot_nominal_vs_real_price(summary_df: pd.DataFrame):
    """Plots trend of Nominal vs. Real average prices."""
    years = summary_df["clean_year"].values
    mean_nom_billion = summary_df["mean_nominal"].values / 1e9
    mean_real_billion = summary_df["mean_real"].values / 1e9

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        years,
        mean_nom_billion,
        marker="o",
        linewidth=2.5,
        color="#d95f02",
        label="Nominal Price",
    )
    ax.plot(
        years,
        mean_real_billion,
        marker="s",
        linewidth=2.5,
        color="#2b5c8f",
        label="Real Price (Base 1400)",
    )

    for x, yn, yr in zip(years, mean_nom_billion, mean_real_billion):
        ax.annotate(
            f"{yn:.2f}B",
            (x, yn),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold",
            color="#d95f02",
        )
        ax.annotate(
            f"{yr:.2f}B",
            (x, yr),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontweight="bold",
            color="#2b5c8f",
        )

    ax.set_title(
        "Nominal vs. Inflation-Adjusted (Real) Average Price (1400-1403)",
        fontsize=13,
        pad=15,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Price (Billion Tomans)")
    ax.set_xticks(years)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_category_price_distribution(df: pd.DataFrame, max_percentile: float = 0.95):
    """Plots price distribution for level-3 categories with outlier trimming."""
    df_plot = df[df["property_type"] != "unknown"].copy()

    # Trim outliers per category to keep boxplot readable
    def trim_group(group):
        upper_limit = group["price_value"].quantile(max_percentile)
        return group[group["price_value"] <= upper_limit]

    df_trimmed = (
        df_plot.groupby("property_type", group_keys=False)
        .apply(trim_group)
        .reset_index(drop=True)
    )
    df_trimmed["price_billion"] = df_trimmed["price_value"] / 1e9

    plt.figure(figsize=(12, 6))
    order = (
        df_trimmed.groupby("property_type")["price_billion"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(
        data=df_trimmed,
        x="property_type",
        y="price_billion",
        order=order,
        palette="Set2",
    )
    plt.title(
        f"Sale Price Distribution by Level-3 Category (Trimmed at {int(max_percentile*100)}th Percentile)",
        fontsize=13,
    )
    plt.xlabel("Property Category")
    plt.ylabel("Price (Billion Tomans)")
    plt.xticks(rotation=30)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
