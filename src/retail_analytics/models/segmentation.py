"""Store segmentation via K-Means."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import PowerTransformer, StandardScaler

from retail_analytics.config import SegmentationConfig, get_config


@dataclass
class SegmentationResult:
    store_profiles: pd.DataFrame
    model: KMeans
    power_transformer: PowerTransformer
    scaler: StandardScaler
    pca: PCA
    silhouette: float
    pca_variance_explained: float
    wcss: list[float]


def build_store_profiles(master_df: pd.DataFrame) -> pd.DataFrame:
    return (
        master_df.groupby("Store")
        .agg(
            Weekly_Sales=("Weekly_Sales", "mean"),
            Size=("Size", "first"),
            Total_MD_Value=("Total_MD_Value", "mean"),
            Unemployment=("Unemployment", "mean"),
        )
        .reset_index()
    )


def fit_segmentation(
    master_df: pd.DataFrame,
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    config = config or get_config().segmentation
    profiles = build_store_profiles(master_df)

    feature_matrix = profiles.drop(columns=["Store"])
    power_transformer = PowerTransformer(method="yeo-johnson")
    scaler = StandardScaler()
    transformed = power_transformer.fit_transform(feature_matrix)
    scaled = scaler.fit_transform(transformed)

    pca = PCA(n_components=config.pca_components, random_state=config.random_state)
    pca_features = pca.fit_transform(scaled)

    wcss: list[float] = []
    for k in range(1, config.elbow_max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", random_state=config.random_state, n_init=10)
        km.fit(pca_features)
        wcss.append(float(km.inertia_))

    kmeans = KMeans(
        n_clusters=config.n_clusters,
        init="k-means++",
        random_state=config.random_state,
        n_init=10,
    )
    profiles = profiles.copy()
    profiles["Cluster"] = kmeans.fit_predict(pca_features)

    silhouette = float("nan")
    n_labels = profiles["Cluster"].nunique()
    if len(pca_features) > n_labels and n_labels > 1:
        silhouette = float(silhouette_score(pca_features, profiles["Cluster"]))

    return SegmentationResult(
        store_profiles=profiles,
        model=kmeans,
        power_transformer=power_transformer,
        scaler=scaler,
        pca=pca,
        silhouette=silhouette,
        pca_variance_explained=float(pca.explained_variance_ratio_.sum()),
        wcss=wcss,
    )


def attach_clusters(master_df: pd.DataFrame, store_profiles: pd.DataFrame) -> pd.DataFrame:
    return master_df.merge(store_profiles[["Store", "Cluster"]], on="Store", how="left")


def cluster_summary(store_profiles: pd.DataFrame) -> pd.DataFrame:
    return (
        store_profiles.groupby("Cluster")
        .agg(
            store_count=("Store", "count"),
            avg_weekly_sales=("Weekly_Sales", "mean"),
            avg_size=("Size", "mean"),
            avg_unemployment=("Unemployment", "mean"),
        )
        .round(2)
    )
