"""Configuration loading and environment overrides."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class DataConfig(BaseModel):
    sales_file: str
    stores_file: str
    features_file: str
    date_format: str = "%d/%m/%Y"
    test_split_date: str = "2012-01-01"


class QualityConfig(BaseModel):
    min_stores: int = 40
    min_rows: int = 400_000
    max_negative_sales_pct: float = 0.01
    required_sales_columns: list[str]
    required_stores_columns: list[str]
    required_features_columns: list[str]


class FeaturesConfig(BaseModel):
    markdown_columns: list[str]
    temp_bins: dict[str, float]


class SegmentationConfig(BaseModel):
    n_clusters: int = 4
    pca_components: int = 2
    random_state: int = 42
    elbow_max_k: int = 10


class XGBoostConfig(BaseModel):
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 7
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    random_state: int = 42


class ForecastingConfig(BaseModel):
    model_type: str = "xgboost"
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    feature_columns: list[str]


class AnomalyConfig(BaseModel):
    zscore_threshold: float = 3.0


class AssociationConfig(BaseModel):
    min_support: float = 0.011
    min_confidence: float = 0.8
    min_lift: float = 3.0
    min_length: int = 2


class SarimaConfig(BaseModel):
    seasonal_period: int = 52
    stepwise: bool = True


class ArtifactsConfig(BaseModel):
    champion_model_name: str = "xgboost_champion"
    segmentation_name: str = "store_segments"


class PipelineConfig(BaseModel):
    data: DataConfig
    quality: QualityConfig
    features: FeaturesConfig
    segmentation: SegmentationConfig
    forecasting: ForecastingConfig
    anomaly: AnomalyConfig
    association: AssociationConfig
    sarima: SarimaConfig
    artifacts: ArtifactsConfig


class Settings(BaseSettings):
    """Runtime settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_",
        env_file=".env",
        extra="ignore",
    )

    config_path: Path = DEFAULT_CONFIG_PATH
    data_dir: Path = PROJECT_ROOT / "data" / "raw" / "Retail Datsets"
    raw_data_dir: Path | None = None
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    outputs_dir: Path = PROJECT_ROOT / "data" / "outputs"
    log_level: str = "INFO"
    random_seed: int = 42

    @property
    def source_data_dir(self) -> Path:
        return self.raw_data_dir or self.data_dir


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config() -> PipelineConfig:
    settings = get_settings()
    raw = load_yaml_config(settings.config_path)
    return PipelineConfig.model_validate(raw)


def ensure_directories(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    for directory in (
        settings.processed_dir,
        settings.artifacts_dir,
        settings.outputs_dir,
        settings.artifacts_dir / "models",
        settings.artifacts_dir / "reports",
    ):
        directory.mkdir(parents=True, exist_ok=True)
