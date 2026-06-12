"""Data ingestion and merging."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_analytics.config import DataConfig, Settings, get_config, get_settings


def _parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.upper()
    return normalized.isin({"TRUE", "1", "T", "YES"})


def load_raw_tables(
    data_dir: Path | None = None,
    data_config: DataConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    config = get_config()
    data_config = data_config or config.data
    data_dir = data_dir or settings.source_data_dir

    sales = pd.read_csv(data_dir / data_config.sales_file)
    stores = pd.read_csv(data_dir / data_config.stores_file)
    features = pd.read_csv(data_dir / data_config.features_file)

    sales["Date"] = pd.to_datetime(sales["Date"], dayfirst=True)
    features["Date"] = pd.to_datetime(features["Date"], dayfirst=True)
    sales["IsHoliday"] = _parse_bool(sales["IsHoliday"])
    features["IsHoliday"] = _parse_bool(features["IsHoliday"])

    return sales, stores, features


def merge_datasets(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    master = sales.merge(stores, on="Store", how="left")
    master = master.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    return master


def clean_master_df(
    master_df: pd.DataFrame,
    markdown_columns: list[str] | None = None,
    max_negative_sales_pct: float = 0.01,
) -> pd.DataFrame:
    config = get_config()
    markdown_columns = markdown_columns or config.features.markdown_columns
    df = master_df.copy()

    negative_mask = df["Weekly_Sales"] < 0
    negative_pct = negative_mask.mean()
    if negative_pct > max_negative_sales_pct:
        raise ValueError(
            f"Negative sales exceed threshold: {negative_pct:.2%} > {max_negative_sales_pct:.2%}"
        )
    df = df.loc[~negative_mask].copy()

    df[markdown_columns] = df[markdown_columns].fillna(0)
    df["CPI"] = df.groupby("Store")["CPI"].ffill()
    df["Unemployment"] = df.groupby("Store")["Unemployment"].ffill()
    return df


def ingest(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    sales, stores, features = load_raw_tables(settings.source_data_dir)
    master = merge_datasets(sales, stores, features)
    return clean_master_df(master)
