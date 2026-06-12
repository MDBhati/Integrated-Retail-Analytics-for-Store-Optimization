"""Tests for data quality validation."""

from retail_analytics.data.quality import validate_master_df, validate_raw_tables


def test_validate_raw_tables_passes_on_real_data(settings):
    from retail_analytics.data.load import load_raw_tables

    sales, stores, features = load_raw_tables(settings.source_data_dir)
    result = validate_raw_tables(sales, stores, features)
    assert result.passed, result.errors


def test_validate_raw_tables_detects_missing_columns(sample_sales, sample_stores, sample_features):
    broken = sample_sales.drop(columns=["Weekly_Sales"])
    result = validate_raw_tables(broken, sample_stores, sample_features)
    assert not result.passed
    assert any("Weekly_Sales" in err for err in result.errors)


def test_validate_master_df(sample_sales, sample_stores, sample_features):
    from retail_analytics.data.load import clean_master_df, merge_datasets

    merged = merge_datasets(sample_sales, sample_stores, sample_features)
    master = clean_master_df(merged, max_negative_sales_pct=1.0)
    result = validate_master_df(master)
    assert result.passed
