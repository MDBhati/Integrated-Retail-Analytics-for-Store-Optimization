"""Tests for store segmentation."""

from retail_analytics.config import SegmentationConfig
from retail_analytics.data.load import clean_master_df, merge_datasets
from retail_analytics.features.engineering import engineer_features
from retail_analytics.models.segmentation import attach_clusters, fit_segmentation


def test_segmentation_assigns_clusters(sample_sales, sample_stores, sample_features):
    master = engineer_features(
        clean_master_df(
            merge_datasets(sample_sales, sample_stores, sample_features),
            max_negative_sales_pct=1.0,
        )
    )
    seg_config = SegmentationConfig(n_clusters=2, elbow_max_k=2)
    result = fit_segmentation(master, seg_config)
    assert len(result.store_profiles) == master["Store"].nunique()
    assert "Cluster" in result.store_profiles.columns

    enriched = attach_clusters(master, result.store_profiles)
    assert enriched["Cluster"].notna().all()
