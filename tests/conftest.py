"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retail_analytics.config import Settings


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(project_root: Path) -> Settings:
    return Settings(
        data_dir=project_root / "data" / "raw" / "Retail Datsets",
        processed_dir=project_root / "data" / "processed",
        artifacts_dir=project_root / "artifacts",
        outputs_dir=project_root / "data" / "outputs",
        log_level="WARNING",
    )


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [1, 1, 1, 2, 2, 2],
            "Dept": [1, 2, 1, 1, 2, 1],
            "Date": pd.to_datetime(
                ["2010-02-05", "2010-02-05", "2010-02-12", "2010-02-05", "2010-02-05", "2010-02-12"]
            ),
            "Weekly_Sales": [100.0, 80.0, 120.0, 90.0, 70.0, 110.0],
            "IsHoliday": [False, False, True, False, False, True],
        }
    )


@pytest.fixture
def sample_stores() -> pd.DataFrame:
    return pd.DataFrame({"Store": [1, 2], "Type": ["A", "B"], "Size": [150000, 40000]})


@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "Date": pd.to_datetime(["2010-02-05", "2010-02-12", "2010-02-05", "2010-02-12"]),
            "Temperature": [40.0, 38.0, 45.0, 42.0],
            "Fuel_Price": [2.5, 2.5, 2.6, 2.6],
            "MarkDown1": [0, 5, 0, 0],
            "MarkDown2": [0, 0, 0, 0],
            "MarkDown3": [0, 0, 0, 0],
            "MarkDown4": [0, 0, 0, 0],
            "MarkDown5": [0, 0, 0, 0],
            "CPI": [210.0, 210.5, 211.0, 211.5],
            "Unemployment": [8.0, 8.0, 7.5, 7.5],
            "IsHoliday": [False, True, False, True],
        }
    )
