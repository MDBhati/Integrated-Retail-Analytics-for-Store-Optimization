"""Feature engineering pipeline."""

from __future__ import annotations

import pandas as pd

from retail_analytics.config import FeaturesConfig, get_config


def categorize_temperature(
    temperature: float,
    bins: dict[str, float] | None = None,
) -> str:
    bins = bins or get_config().features.temp_bins
    if pd.isna(temperature):
        return "Unknown"
    if temperature <= bins["freezing"]:
        return "Freezing"
    if temperature <= bins["cold"]:
        return "Cold"
    if temperature <= bins["moderate"]:
        return "Moderate"
    return "Hot"


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Week"] = out["Date"].dt.isocalendar().week.astype(int)
    return out


def add_retail_signals(
    df: pd.DataFrame,
    markdown_columns: list[str] | None = None,
) -> pd.DataFrame:
    config = get_config()
    markdown_columns = markdown_columns or config.features.markdown_columns
    out = df.sort_values(["Store", "Dept", "Date"]).copy()
    out["Sales_Lag_1"] = (
        out.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).fillna(0)
    )
    md = out[markdown_columns]
    out["MD_Count"] = (md > 0).sum(axis=1)
    out["Total_MD_Value"] = md.sum(axis=1)
    return out


def engineer_features(
    df: pd.DataFrame,
    features_config: FeaturesConfig | None = None,
) -> pd.DataFrame:
    features_config = features_config or get_config().features
    out = add_temporal_features(df)
    out["Temp_Category"] = out["Temperature"].apply(
        lambda t: categorize_temperature(t, features_config.temp_bins)
    )
    out = add_retail_signals(out, features_config.markdown_columns)
    return out


def encode_store_type(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    mapping = {label: idx for idx, label in enumerate(sorted(df["Type"].dropna().unique()))}
    out = df.copy()
    out["Type_Encoded"] = out["Type"].map(mapping)
    return out, mapping
