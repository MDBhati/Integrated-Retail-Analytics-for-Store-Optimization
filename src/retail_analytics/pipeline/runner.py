"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from retail_analytics.anomaly.detection import anomaly_summary, flag_anomalies
from retail_analytics.artifacts import new_run_id, save_artifact, write_json_report, write_manifest
from retail_analytics.config import Settings, ensure_directories, get_config, get_settings
from retail_analytics.data.load import clean_master_df, ingest, load_raw_tables, merge_datasets
from retail_analytics.data.quality import validate_master_df, validate_raw_tables
from retail_analytics.features.engineering import engineer_features
from retail_analytics.logging_setup import setup_logging
from retail_analytics.models.association import mine_association_rules
from retail_analytics.models.forecasting import train_forecaster
from retail_analytics.models.sarima import fit_sarima_baseline
from retail_analytics.models.segmentation import attach_clusters, cluster_summary, fit_segmentation


@dataclass
class PipelineResult:
    run_id: str
    master_df: pd.DataFrame
    forecasts: pd.DataFrame
    store_profiles: pd.DataFrame
    association_rules: pd.DataFrame
    reports: dict


def run_training_pipeline(settings: Settings | None = None) -> PipelineResult:
    settings = settings or get_settings()
    config = get_config()
    ensure_directories(settings)
    logger = setup_logging(settings.log_level)
    run_id = new_run_id()
    logger.info("Starting training pipeline run_id=%s", run_id)

    sales, stores, features = load_raw_tables(settings.source_data_dir)
    quality = validate_raw_tables(sales, stores, features)
    quality.raise_if_failed()
    logger.info("Raw data quality checks passed: %s", quality.checks)

    master = clean_master_df(merge_datasets(sales, stores, features))
    master_quality = validate_master_df(master)
    master_quality.raise_if_failed()

    master = engineer_features(master)
    master = flag_anomalies(master)
    logger.info("Feature engineering and anomaly detection complete")

    seg = fit_segmentation(master)
    master = attach_clusters(master, seg.store_profiles)
    logger.info("Store segmentation complete (silhouette=%.2f)", seg.silhouette)

    forecast_artifacts, forecasts = train_forecaster(master)
    logger.info(
        "Forecast training complete (R2=%.4f, MAE=%.2f)",
        forecast_artifacts.metrics["r2"],
        forecast_artifacts.metrics["mae"],
    )

    sarima = fit_sarima_baseline(sales)
    logger.info("SARIMA baseline fit complete: %sx%s", sarima.order, sarima.seasonal_order)

    rules = mine_association_rules(sales)
    logger.info("Association rule mining complete (%d rules)", len(rules))

    processed_path = settings.processed_dir / f"master_{run_id}.parquet"
    master.to_parquet(processed_path, index=False)
    forecasts_path = settings.outputs_dir / f"forecasts_{run_id}.parquet"
    forecasts.to_parquet(forecasts_path, index=False)
    rules_path = settings.outputs_dir / f"association_rules_{run_id}.parquet"
    rules.to_parquet(rules_path, index=False)

    model_bundle = {
        "forecast_model": forecast_artifacts.model,
        "feature_columns": forecast_artifacts.feature_columns,
        "type_encoding": forecast_artifacts.type_encoding,
        "segmentation": {
            "kmeans": seg.model,
            "power_transformer": seg.power_transformer,
            "scaler": seg.scaler,
            "pca": seg.pca,
        },
        "store_profiles": seg.store_profiles,
    }
    champion = config.artifacts.champion_model_name
    model_path = settings.artifacts_dir / "models" / f"{champion}.joblib"
    save_artifact(model_bundle, model_path)

    reports = {
        "forecast_metrics": forecast_artifacts.metrics,
        "segmentation": {
            "silhouette": seg.silhouette,
            "pca_variance_explained": seg.pca_variance_explained,
            "cluster_summary": cluster_summary(seg.store_profiles).to_dict(),
        },
        "anomaly_summary": anomaly_summary(master),
        "sarima": {
            "order": sarima.order,
            "seasonal_order": sarima.seasonal_order,
        },
        "association_rule_count": len(rules),
    }
    report_path = settings.artifacts_dir / "reports" / f"evaluation_{run_id}.json"
    write_json_report(report_path, reports)

    write_manifest(
        settings.artifacts_dir / "manifests" / f"{run_id}.json",
        run_id=run_id,
        stage="train",
        metadata={
            "processed_path": str(processed_path),
            "forecasts_path": str(forecasts_path),
            "model_path": str(model_path),
            "report_path": str(report_path),
            "reports": reports,
        },
    )

    metrics = reports["forecast_metrics"]
    logger.info("Training complete. R2=%.4f MAE=%.2f", metrics["r2"], metrics["mae"])
    return PipelineResult(
        run_id=run_id,
        master_df=master,
        forecasts=forecasts,
        store_profiles=seg.store_profiles,
        association_rules=rules,
        reports=reports,
    )


def run_scoring_pipeline(
    settings: Settings | None = None,
    model_path: Path | None = None,
) -> pd.DataFrame:
    settings = settings or get_settings()
    config = get_config()
    logger = setup_logging(settings.log_level)
    run_id = new_run_id()

    from retail_analytics.artifacts import load_artifact
    from retail_analytics.models.forecasting import score_forecasts

    champion = config.artifacts.champion_model_name
    default_model = settings.artifacts_dir / "models" / f"{champion}.joblib"
    model_path = model_path or default_model
    bundle = load_artifact(model_path)

    master = engineer_features(ingest(settings))
    master = attach_clusters(master, bundle["store_profiles"])
    scored = score_forecasts(
        bundle["forecast_model"],
        master,
        bundle["feature_columns"],
        bundle["type_encoding"],
    )

    output_path = settings.outputs_dir / f"batch_scores_{run_id}.parquet"
    score_cols = ["Store", "Dept", "Date", "Weekly_Sales", "point_forecast"]
    scored[score_cols].to_parquet(output_path, index=False)
    logger.info("Scoring complete: %s rows -> %s", len(scored), output_path)
    return scored
