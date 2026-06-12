# Runbooks

Operational procedures for the retail analytics batch pipeline.

## Pipeline failure: data quality gate

**Symptoms:** `validate-data` or training exits with schema/row-count errors.

**Steps:**
1. Confirm upstream CSV feeds landed in `RETAIL_DATA_DIR` (default: `Retail Datsets/`).
2. Run `retail-analytics validate-data` and note failing checks.
3. Compare row counts and column names against `config/config.yaml` contracts.
4. If duplicate Store–Dept–Date keys appear, dedupe at source or open a data ticket.
5. Re-run `retail-analytics run-pipeline` after fix.

## Pipeline failure: late or missing data

**Symptoms:** Job succeeds but forecasts are stale; planning cutoff missed.

**Steps:**
1. Check file modification times on raw CSVs.
2. Verify scheduler (cron/Airflow) ran and container had network/storage access.
3. Re-trigger batch job; confirm new `run_id` in `artifacts/manifests/`.

## Forecast accuracy degradation

**Symptoms:** Evaluation report shows sustained MAPE/MAE increase vs. champion.

**Steps:**
1. Open latest `artifacts/reports/evaluation_*.json`.
2. Compare metrics to prior runs; segment by cluster if available.
3. Check for macro series drift (CPI, unemployment) and holiday calendar changes.
4. Train challenger with updated window; promote only if holdout metrics improve.
5. Roll back to prior model artifact in `artifacts/models/` if needed.

## Model rollback

**Steps:**
1. Identify last good model bundle: `artifacts/models/xgboost_champion.joblib`.
2. Restore from backup/registry (copy known-good artifact over champion path).
3. Run `retail-analytics score` to regenerate batch outputs.
4. Notify downstream consumers of `model_version` / manifest `run_id` change.

## Anomaly alert spike

**Symptoms:** Sudden increase in `Is_Anomaly` flags.

**Steps:**
1. Compare `holiday_share_among_anomalies` in evaluation report.
2. If low holiday share, suspect data quality (duplicates, unit changes).
3. If high holiday share, likely true calendar events—suppress in modeling if policy requires.
