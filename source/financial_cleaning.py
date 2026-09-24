import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Constants & Business Thresholds
# ---------------------------------------------------------
# Iranian real estate conversion convention (3% monthly rule):
# Every 1,000,000 Tomans deposit (credit) ≈ 30,000 Tomans monthly rent
RENT_PER_CREDIT_RATE = 30_000

# Upper/lower bounds to eliminate input blunders
IMPOSSIBLE_PRICE_MAX = 1e13  # 10,000 Billion Tomans cap
CREDIT_MAX = 5e11  # 500 Billion Tomans cap
RENT_MAX = 1e10  # 10 Billion Tomans cap
PRICE_MIN = 10_000_000  # Minimum realistic purchase price (10M Tomans)
CREDIT_MIN = 1  # Eliminates zero or negative deposits
WINSOR_HI = 0.01  # Top 1st percentile clipping cap

# Redundant columns (excluding those needed for downstream feature flags)
DEAD_COLS = [
    "rent_type",
    "transformable_price",
    "transformable_credit",
    "transformable_rent",
    "transformed_credit",
    "transformed_rent",
]


def fix_mojibake(val):
    """Repair Persian text corrupted by legacy Windows-1256 decoding issues."""
    if pd.isna(val) or not isinstance(val, str):
        return val
    try:
        return val.encode("cp1256").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return val


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply character encoding repairs across all object/string columns."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(fix_mojibake)
    return df


def add_deal_type(df: pd.DataFrame) -> pd.DataFrame:
    """Classify listing transaction type into 'sale', 'rent', or 'other'."""
    df = df.copy()
    conditions = [
        df["price_value"].notna()
        | df.get("price_mode", pd.Series(dtype=object)).eq("توافقی"),
        df["credit_value"].notna()
        | df["rent_value"].notna()
        | df.get("rent_mode", pd.Series(dtype=object)).eq("توافقی")
        | df.get("credit_mode", pd.Series(dtype=object)).eq("توافقی"),
    ]
    df["deal_type"] = np.select(conditions, ["sale", "rent"], default="other")
    return df


def add_business_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive domain indicators: is_negotiable, is_transformable, is_full_mortgage."""
    df = df.copy()

    df["is_negotiable"] = (
        df.get("price_mode", pd.Series(dtype=object)).eq("توافقی")
        | df.get("rent_mode", pd.Series(dtype=object)).eq("توافقی")
        | df.get("credit_mode", pd.Series(dtype=object)).eq("توافقی")
    )

    df["is_transformable"] = (
        df["rent_credit_transform"].fillna(False).astype(bool)
        if "rent_credit_transform" in df.columns
        else False
    )

    rent = pd.to_numeric(df.get("rent_value"), errors="coerce")
    df["is_full_mortgage"] = (df["deal_type"] == "rent") & (rent.isna() | rent.eq(0))
    return df


def remove_impossible_values(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Mask impossible financial bounds to NaN based on domain boundaries."""
    df = df.copy()
    report = {}
    rules = {
        "price_value": (PRICE_MIN, IMPOSSIBLE_PRICE_MAX),
        "credit_value": (CREDIT_MIN, CREDIT_MAX),
        "rent_value": (0, RENT_MAX),
    }

    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad = s.notna() & ~s.between(lo, hi)
        if bad.sum():
            report[col] = int(bad.sum())
        df[col] = s.where(~bad, np.nan)

    if verbose and report:
        print("[Data Cleaning] Impossible values mapped to NaN:")
        for k, v in report.items():
            print(f"  - {k}: {v:,} rows")
    return df


def winsorize_targets(
    df: pd.DataFrame, cols=None, upper: float = WINSOR_HI, verbose: bool = True
) -> pd.DataFrame:
    """Cap extreme outliers at the (1 - upper) quantile."""
    df = df.copy()
    if cols is None:
        cols = [
            c for c in ("price_value", "credit_value", "rent_value") if c in df.columns
        ]

    for col in cols:
        s = df[col]
        cap = s.quantile(1 - upper)
        cut = s > cap
        if cut.sum():
            df[col] = s.where(~cut, cap)
            if verbose:
                print(
                    f"  - Winsorized {col}: {int(cut.sum()):,} rows capped at {cap:,.0f}"
                )
    return df


def add_full_credit_equivalent(df: pd.DataFrame) -> pd.DataFrame:
    """Compute unified full deposit equivalent for rental listings based on market rate."""
    df = df.copy()
    df["equivalent_full_credit"] = np.nan

    mask = (
        df["deal_type"].eq("rent")
        & ~df["is_negotiable"]
        & df["credit_value"].notna()
        & df["rent_value"].notna()
    )
    df.loc[mask, "equivalent_full_credit"] = (
        df.loc[mask, "credit_value"]
        + (df.loc[mask, "rent_value"] / RENT_PER_CREDIT_RATE) * 1_000_000
    )
    return df


def drop_dead_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop low-variance, near-empty, or redundant feature columns."""
    df = df.copy()
    existing = [c for c in DEAD_COLS if c in df.columns]
    if verbose and existing:
        print(f"[Data Cleaning] Dropping dead/redundant columns: {existing}")
    return df.drop(columns=existing)


def split_datasets(df: pd.DataFrame):
    """Split processed data into Analytical and Training subsets."""
    df_sale = df[df["deal_type"] == "sale"].copy()
    df_rent = df[df["deal_type"] == "rent"].copy()

    train_sale = df_sale[
        df_sale["price_value"].notna() & ~df_sale["is_negotiable"]
    ].copy()
    train_rent = df_rent[
        (df_rent["credit_value"].notna() | df_rent["rent_value"].notna())
        & ~df_rent["is_negotiable"]
    ].copy()

    return df_sale, df_rent, train_sale, train_rent


def clean_financial_pipeline(
    df: pd.DataFrame, verbose: bool = True, winsorize: bool = True
):
    """Execute end-to-end financial data cleaning pipeline."""
    df = clean_text_columns(df)
    df = add_deal_type(df)
    df = add_business_flags(df)
    df = remove_impossible_values(df, verbose=verbose)
    if winsorize:
        df = winsorize_targets(df, verbose=verbose)
    df = add_full_credit_equivalent(df)

    df_daily = (
        df[df["rent_price_on_regular_days"].notna()].copy()
        if "rent_price_on_regular_days" in df.columns
        else df.iloc[0:0].copy()
    )

    df = drop_dead_columns(df, verbose=verbose)
    return df, df_daily
