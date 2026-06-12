"""Data quality validation gates."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from retail_analytics.config import QualityConfig, get_config


@dataclass
class ValidationResult:
    passed: bool
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            msg = "Data quality checks failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(msg)


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def validate_raw_tables(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    features: pd.DataFrame,
    quality: QualityConfig | None = None,
) -> ValidationResult:
    quality = quality or get_config().quality
    result = ValidationResult(passed=True)

    for name, df, required in (
        ("sales", sales, quality.required_sales_columns),
        ("stores", stores, quality.required_stores_columns),
        ("features", features, quality.required_features_columns),
    ):
        missing = _missing_columns(df, required)
        if missing:
            result.passed = False
            result.errors.append(f"{name}: missing columns {missing}")
        else:
            result.checks.append(f"{name}: schema OK ({len(df):,} rows)")

    if len(stores) < quality.min_stores:
        result.passed = False
        result.errors.append(f"stores: count {len(stores)} < min {quality.min_stores}")

    if len(sales) < quality.min_rows:
        result.passed = False
        result.errors.append(f"sales: count {len(sales):,} < min {quality.min_rows:,}")

    dup_sales = sales.duplicated(subset=["Store", "Dept", "Date"], keep=False).sum()
    if dup_sales:
        result.passed = False
        result.errors.append(f"sales: {dup_sales} duplicate Store-Dept-Date keys")

    orphan_stores = set(sales["Store"].unique()) - set(stores["Store"].unique())
    if orphan_stores:
        result.passed = False
        result.errors.append(f"sales: {len(orphan_stores)} stores missing from stores table")

    neg_pct = 0.0
    if "Weekly_Sales" in sales.columns:
        neg_pct = (sales["Weekly_Sales"] < 0).mean()
    if neg_pct > quality.max_negative_sales_pct:
        result.errors.append(
            f"sales: negative Weekly_Sales {neg_pct:.2%} exceeds "
            f"{quality.max_negative_sales_pct:.2%}"
        )
        result.checks.append("sales: negative rows will be filtered during cleaning")

    return result


def validate_master_df(master_df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult(passed=True)
    required = {"Store", "Dept", "Date", "Weekly_Sales", "Type", "Size", "Temperature"}
    missing = required - set(master_df.columns)
    if missing:
        result.passed = False
        result.errors.append(f"master: missing columns {sorted(missing)}")
    else:
        result.checks.append(f"master: {len(master_df):,} rows after merge")

    null_pct = master_df[list(required)].isnull().mean()
    high_null = null_pct[null_pct > 0.05]
    if not high_null.empty:
        result.passed = False
        result.errors.append(f"master: high null rates in {high_null.to_dict()}")

    return result
