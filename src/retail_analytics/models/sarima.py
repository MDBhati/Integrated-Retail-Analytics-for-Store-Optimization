"""SARIMA statistical baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from retail_analytics.config import SarimaConfig, get_config


@dataclass
class SarimaResult:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    fitted_series: pd.Series
    model_summary: str


def aggregate_weekly_sales(sales: pd.DataFrame) -> pd.Series:
    total = sales.groupby("Date")["Weekly_Sales"].sum()
    total.index = pd.DatetimeIndex(total.index)
    return total.sort_index()


def fit_sarima_baseline(
    sales: pd.DataFrame,
    config: SarimaConfig | None = None,
) -> SarimaResult:
    config = config or get_config().sarima
    ts_data = aggregate_weekly_sales(sales)

    auto_model = pm.auto_arima(
        ts_data,
        seasonal=True,
        m=config.seasonal_period,
        stepwise=config.stepwise,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        suppress_warnings=True,
        error_action="ignore",
    )
    model = SARIMAX(
        ts_data,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
    ).fit(disp=False)

    fitted = model.fittedvalues
    return SarimaResult(
        order=tuple(auto_model.order),
        seasonal_order=tuple(auto_model.seasonal_order),
        fitted_series=fitted,
        model_summary=str(model.summary()),
    )
