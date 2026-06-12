"""Integration tests for pipeline stages."""

import pytest

from retail_analytics.config import SegmentationConfig
from retail_analytics.models.forecasting import train_forecaster
from retail_analytics.pipeline.runner import run_training_pipeline


@pytest.mark.slow
def test_full_training_pipeline(settings):
    result = run_training_pipeline(settings)
    assert result.run_id
    assert result.reports["forecast_metrics"]["r2"] > 0.5
    assert len(result.store_profiles) >= 40
    assert len(result.forecasts) > 0


def test_forecaster_on_subset(settings):
    from retail_analytics.data.load import ingest
    from retail_analytics.features.engineering import engineer_features
    from retail_analytics.models.segmentation import attach_clusters, fit_segmentation

    master = engineer_features(ingest(settings))
    master = master[master["Date"] < "2011-06-01"].copy()
    seg = fit_segmentation(master, SegmentationConfig(n_clusters=2, elbow_max_k=3))
    master = attach_clusters(master, seg.store_profiles)

    artifacts, test_df = train_forecaster(master, split_date="2011-01-01")
    assert artifacts.metrics["mae"] >= 0
    assert len(test_df) > 0
