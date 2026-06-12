# Output schemas

## Batch forecasts (`data/outputs/forecasts_*.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| Store | int | Store identifier |
| Dept | int | Department identifier |
| Date | datetime | Week ending date |
| Weekly_Sales | float | Actual sales (holdout period) |
| predicted_weekly_sales | float | Model point forecast |

## Batch scores (`data/outputs/batch_scores_*.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| Store | int | Store identifier |
| Dept | int | Department identifier |
| Date | datetime | Week ending date |
| Weekly_Sales | float | Actual sales |
| point_forecast | float | Scored forecast |

## Store segments (`artifacts/models/xgboost_champion.joblib` → `store_profiles`)

| Column | Type | Description |
|--------|------|-------------|
| Store | int | Store identifier |
| Cluster | int | Segment cluster ID (0–3) |
| Weekly_Sales | float | Average weekly sales |
| Size | float | Store footprint |
| Total_MD_Value | float | Average markdown spend |
| Unemployment | float | Average local unemployment |

## Run manifest (`artifacts/manifests/{run_id}.json`)

| Field | Description |
|-------|-------------|
| run_id | Unique pipeline run identifier |
| stage | `train` or `score` |
| created_at | UTC timestamp |
| processed_path | Path to curated master table |
| model_path | Champion model artifact |
| report_path | Evaluation metrics JSON |

## Evaluation report (`artifacts/reports/evaluation_{run_id}.json`)

Contains `forecast_metrics` (r2, mae, rmse, mape), `segmentation` summary, `anomaly_summary`, SARIMA order, and association rule count.
