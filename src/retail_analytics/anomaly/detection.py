"""Sales anomaly detection."""

from __future__ import annotations

import pandas as pd

from retail_analytics.config import get_config


def flag_anomalies(
    df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    config = get_config()
    threshold = threshold if threshold is not None else config.anomaly.zscore_threshold
    out = df.copy()
    out["Sales_ZScore"] = out.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    out["Is_Anomaly"] = out["Sales_ZScore"].abs() > threshold
    return out


def anomaly_summary(df: pd.DataFrame) -> dict:
    flagged = int(df["Is_Anomaly"].sum()) if "Is_Anomaly" in df.columns else 0
    holiday_share = 0.0
    if flagged and "IsHoliday" in df.columns:
        holiday_share = float(df.loc[df["Is_Anomaly"], "IsHoliday"].mean())
    return {
        "total_anomalies": flagged,
        "anomaly_rate": float(flagged / len(df)) if len(df) else 0.0,
        "holiday_share_among_anomalies": holiday_share,
    }
