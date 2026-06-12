"""XGBoost demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import xgboost as xgb

from retail_analytics.config import ForecastingConfig, get_config
from retail_analytics.features.engineering import encode_store_type
from retail_analytics.metrics import regression_metrics


@dataclass
class ForecastArtifacts:
    model: xgb.XGBRegressor
    feature_columns: list[str]
    type_encoding: dict[str, int]
    train_end_date: str
    metrics: dict[str, float]


def prepare_model_frame(master_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    config = get_config()
    df, type_encoding = encode_store_type(master_df)
    df[config.features.markdown_columns] = df[config.features.markdown_columns].fillna(0)
    return df, type_encoding


def time_series_split(
    df: pd.DataFrame,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(split_date)
    train = df[df["Date"] < cutoff].copy()
    test = df[df["Date"] >= cutoff].copy()
    return train, test


def train_forecaster(
    master_df: pd.DataFrame,
    config: ForecastingConfig | None = None,
    split_date: str | None = None,
) -> tuple[ForecastArtifacts, pd.DataFrame]:
    config = config or get_config().forecasting
    split_date = split_date or get_config().data.test_split_date
    df, type_encoding = prepare_model_frame(master_df)
    train_df, test_df = time_series_split(df, split_date)

    feature_columns = config.feature_columns
    x_train = train_df[feature_columns]
    y_train = train_df["Weekly_Sales"]
    x_test = test_df[feature_columns]
    y_test = test_df["Weekly_Sales"]

    params = config.xgboost.model_dump()
    early_stopping = params.pop("early_stopping_rounds")
    model = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping, n_jobs=-1)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=False,
    )

    preds = model.predict(x_test)
    metrics = regression_metrics(y_test, preds)

    test_df = test_df.copy()
    test_df["predicted_weekly_sales"] = preds

    artifacts = ForecastArtifacts(
        model=model,
        feature_columns=feature_columns,
        type_encoding=type_encoding,
        train_end_date=split_date,
        metrics=metrics,
    )
    return artifacts, test_df


def score_forecasts(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    feature_columns: list[str],
    type_encoding: dict[str, int],
) -> pd.DataFrame:
    scored = df.copy()
    scored["Type_Encoded"] = scored["Type"].map(type_encoding)
    scored["point_forecast"] = model.predict(scored[feature_columns])
    return scored
