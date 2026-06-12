"""Tests for feature engineering."""


from retail_analytics.data.load import clean_master_df, merge_datasets
from retail_analytics.features.engineering import (
    categorize_temperature,
    engineer_features,
)


def test_categorize_temperature_bins():
    assert categorize_temperature(20) == "Freezing"
    assert categorize_temperature(45) == "Cold"
    assert categorize_temperature(65) == "Moderate"
    assert categorize_temperature(90) == "Hot"


def test_engineer_features_adds_columns(sample_sales, sample_stores, sample_features):
    master = merge_datasets(sample_sales, sample_stores, sample_features)
    master = clean_master_df(master, max_negative_sales_pct=1.0)
    result = engineer_features(master)

    expected = [
        "Year", "Month", "Week", "Temp_Category",
        "Sales_Lag_1", "MD_Count", "Total_MD_Value",
    ]
    for col in expected:
        assert col in result.columns

    assert result.loc[result["Store"].eq(1) & result["Dept"].eq(1), "Sales_Lag_1"].iloc[0] == 0
    assert result["Total_MD_Value"].sum() >= 0


def test_sales_lag_is_per_store_dept(sample_sales, sample_stores, sample_features):
    master = merge_datasets(sample_sales, sample_stores, sample_features)
    master = clean_master_df(master, max_negative_sales_pct=1.0)
    result = engineer_features(master)
    lag_row = result[(result["Store"] == 1) & (result["Dept"] == 1)].sort_values("Date").iloc[1]
    assert lag_row["Sales_Lag_1"] == 100.0
