import numpy as np
import pandas as pd


def add_equivalent_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total financial equivalents using standard Iranian real estate formula:
    1 Million Toman Rent ≈ 30 Million Toman Full Credit (Mortgage) equivalent.
    """
    df = df.copy()

    if "credit_value" in df.columns and "rent_value" in df.columns:
        credit = pd.to_numeric(df["credit_value"], errors="coerce").fillna(0)
        rent = pd.to_numeric(df["rent_value"], errors="coerce").fillna(0)

        # Total Equivalent Credit: Deposit + (Monthly Rent * 30)
        # Only meaningful if at least one of credit or rent is positive
        has_rental_data = (credit > 0) | (rent > 0)
        equiv_credit = credit + (rent * 30.0)
        df["equivalent_full_credit"] = equiv_credit.where(has_rental_data, np.nan)

    return df


def add_rental_daily_pricing_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price multipliers for temporary/daily rentals:
    - Weekend price ratio vs regular days
    - Special days / holiday price ratio vs regular days
    """
    df = df.copy()
    base_col = "rent_price_on_regular_days"

    if base_col in df.columns:
        reg_price = pd.to_numeric(df[base_col], errors="coerce")
        valid_reg = reg_price.where(reg_price > 0, np.nan)

        if "rent_price_at_weekends" in df.columns:
            weekend_price = pd.to_numeric(df["rent_price_at_weekends"], errors="coerce")
            df["weekend_price_ratio"] = weekend_price / valid_reg

        if "rent_price_on_special_days" in df.columns:
            special_price = pd.to_numeric(
                df["rent_price_on_special_days"], errors="coerce"
            )
            df["special_day_price_ratio"] = special_price / valid_reg

    return df


def add_convertibility_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate convertibility columns into unified binary flags.
    """
    df = df.copy()

    # Convertibility between mortgage and rent
    if "rent_credit_transform" in df.columns:
        df["is_rent_credit_convertible"] = (
            df["rent_credit_transform"].notna()
            & (df["rent_credit_transform"] != False)
            & (df["rent_credit_transform"] != 0)
        ).astype(int)

    # Boolean flag indicating bachelor/single tenant allowance
    if "rent_to_single" in df.columns:
        df["allows_single_tenant"] = (
            df["rent_to_single"].notna()
            & (df["rent_to_single"] != False)
            & (df["rent_to_single"] != 0)
        ).astype(int)

    return df


def add_log_transforms(
    df: pd.DataFrame,
    target_cols=(
        "price_value",
        "credit_value",
        "rent_value",
        "equivalent_full_credit",
        "rent_price_on_regular_days",
        "rent_price_at_weekends",
        "rent_price_on_special_days",
    ),
) -> pd.DataFrame:
    """
    Apply log1p transformation to monetary targets to reduce positive skewness.
    """
    df = df.copy()
    for col in target_cols:
        if col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            valid_vals = numeric_vals.where(numeric_vals >= 0, np.nan)
            df[f"log_{col}"] = np.log1p(valid_vals)
    return df


def feature_engineering_pipeline(
    df: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """
    Sequential feature engineering execution on financial attributes.
    """
    df = add_equivalent_financial_metrics(df)
    df = add_rental_daily_pricing_ratios(df)
    df = add_convertibility_flags(df)
    df = add_log_transforms(df)

    if verbose:
        new_cols = [
            c
            for c in [
                "equivalent_full_credit",
                "weekend_price_ratio",
                "special_day_price_ratio",
                "is_rent_credit_convertible",
                "allows_single_tenant",
                "log_price_value",
                "log_rent_value",
                "log_credit_value",
            ]
            if c in df.columns
        ]
        print(
            f"[Financial Feature Engineering] Successfully added features: {new_cols}"
        )

    return df
