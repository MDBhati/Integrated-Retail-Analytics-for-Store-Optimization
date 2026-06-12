"""Command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from retail_analytics.config import ensure_directories, get_settings
from retail_analytics.data.load import ingest, load_raw_tables
from retail_analytics.data.quality import validate_raw_tables
from retail_analytics.logging_setup import setup_logging
from retail_analytics.pipeline.runner import run_scoring_pipeline, run_training_pipeline


@click.group()
@click.option("--log-level", default=None, help="Override RETAIL_LOG_LEVEL")
def main(log_level: str | None) -> None:
    """Retail analytics production CLI."""
    settings = get_settings()
    setup_logging(log_level or settings.log_level)


@main.command("validate-data")
def validate_data() -> None:
    """Run data quality checks on raw inputs."""
    settings = get_settings()
    sales, stores, features = load_raw_tables(settings.source_data_dir)
    result = validate_raw_tables(sales, stores, features)
    for check in result.checks:
        click.echo(f"OK  {check}")
    for error in result.errors:
        click.echo(f"ERR {error}", err=True)
    if not result.passed:
        sys.exit(1)
    click.echo("All data quality checks passed.")


@main.command("ingest")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def ingest_cmd(output: Path | None) -> None:
    """Ingest and clean raw data."""
    settings = get_settings()
    ensure_directories(settings)
    df = ingest(settings)
    output = output or settings.processed_dir / "master_latest.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    click.echo(f"Wrote {len(df):,} rows to {output}")


@main.command("train")
def train_cmd() -> None:
    """Run full training pipeline."""
    result = run_training_pipeline()
    click.echo(f"Training complete. run_id={result.run_id}")
    click.echo(f"Forecast R2: {result.reports['forecast_metrics']['r2']:.4f}")
    click.echo(f"Forecast MAE: {result.reports['forecast_metrics']['mae']:.2f}")


@main.command("score")
@click.option("--model-path", type=click.Path(exists=True, path_type=Path), default=None)
def score_cmd(model_path: Path | None) -> None:
    """Batch score using champion model."""
    scored = run_scoring_pipeline(model_path=model_path)
    click.echo(f"Scored {len(scored):,} rows.")


@main.command("run-pipeline")
def run_pipeline_cmd() -> None:
    """Validate, train, and score in one job."""
    settings = get_settings()
    sales, stores, features = load_raw_tables(settings.source_data_dir)
    validate_raw_tables(sales, stores, features).raise_if_failed()
    run_training_pipeline(settings)
    run_scoring_pipeline(settings)
    click.echo("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
