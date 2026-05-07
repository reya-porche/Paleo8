"""
geo_tvt/clustering/well_clustering.py
Clusters wells by GR pattern + trajectory + TVT behavior.

Why clustering matters:
  Wells probably come from multiple geological sub-environments.
  A single global model averages across all behaviors.
  Per-cluster models capture environment-specific dynamics.

Strategy:
  1. Build a well-level feature vector (GR statistics + trajectory + TVT)
  2. Cluster using K-Means or hierarchical clustering
  3. Assign cluster labels
  4. Train one model per cluster
  5. For test wells, find nearest cluster and use its model

The "similar-well retrieval" idea becomes: find nearest well in embedding space.
"""

import numpy as np
import pandas as pd
from sklearn.cluster         import KMeans, AgglomerativeClustering
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR


def build_well_summary_features(
    df: pd.DataFrame,
    gr_col:   str = "GR",
    md_col:   str = "MD",
    x_col:    str = "X",
    y_col:    str = "Y",
    z_col:    str = "Z",
    tvt_col:  str = "TVT_input",
    well_col: str = "well_id",
) -> pd.DataFrame:
    """
    Reduce each well to a single row of summary features.
    These features describe the well's geological character.

    Returns: DataFrame with one row per well, indexed by well_id.
    """
    rows = []
    for wid, grp in df.groupby(well_col):
        row = {"well_id": wid}

        if gr_col in grp.columns:
            gr = grp[gr_col].dropna()
            if len(gr) > 0:
                row["gr_mean"]    = gr.mean()
                row["gr_std"]     = gr.std()
                row["gr_p10"]     = gr.quantile(0.10)
                row["gr_p25"]     = gr.quantile(0.25)
                row["gr_p50"]     = gr.quantile(0.50)
                row["gr_p75"]     = gr.quantile(0.75)
                row["gr_p90"]     = gr.quantile(0.90)
                row["gr_range"]   = gr.max() - gr.min()
                row["gr_skew"]    = float(gr.skew())
                row["gr_kurt"]    = float(gr.kurtosis())
                row["gr_sand_frac"]  = float((gr < 60).mean())
                row["gr_shale_frac"] = float((gr > 100).mean())
                # Autocorrelation at lag 1 and 5 (cyclicity)
                row["gr_autocorr1"] = float(gr.autocorr(1) or 0)
                row["gr_autocorr5"] = float(gr.autocorr(5) or 0)
                # Trend slope
                x = np.arange(len(gr), dtype=float)
                try:
                    row["gr_trend_slope"] = float(np.polyfit(x, gr.values, 1)[0])
                except Exception:
                    row["gr_trend_slope"] = 0.0

        if z_col in grp.columns:
            z = grp[z_col].dropna()
            if len(z) > 0:
                row["z_range"]  = float(z.max() - z.min())
                row["z_std"]    = float(z.std())
                dz = z.diff().dropna()
                row["z_d1_mean"] = float(dz.mean())
                row["z_d1_std"]  = float(dz.std())

        if x_col in grp.columns and y_col in grp.columns:
            x, y = grp[x_col].dropna(), grp[y_col].dropna()
            if len(x) > 1:
                row["lateral_length_horiz"] = float(np.sqrt(
                    (x.iloc[-1] - x.iloc[0])**2 + (y.iloc[-1] - y.iloc[0])**2
                ))
                row["x_span"] = float(x.max() - x.min())
                row["y_span"] = float(y.max() - y.min())

        if md_col in grp.columns:
            md = grp[md_col].dropna()
            row["md_total"]  = float(md.max() - md.min())
            row["md_n_rows"] = len(md)

        if tvt_col in grp.columns:
            tvt_known = grp[tvt_col].dropna()
            if len(tvt_known) > 5:
                row["tvt_mean"]  = float(tvt_known.mean())
                row["tvt_std"]   = float(tvt_known.std())
                row["tvt_range"] = float(tvt_known.max() - tvt_known.min())
                tvt_d1 = tvt_known.diff().dropna()
                row["tvt_slope_mean"] = float(tvt_d1.mean())
                row["tvt_slope_std"]  = float(tvt_d1.std())
                row["tvt_known_frac"] = float(grp[tvt_col].notna().mean())

        rows.append(row)

    summary = pd.DataFrame(rows).set_index("well_id")
    return summary.fillna(0)


class WellClusterer:
    """
    Clusters wells by geological character and assigns per-cluster labels.
    Supports K-Means and hierarchical clustering.
    """

    def __init__(self, n_clusters: int = 5, method: str = "kmeans", pca_dim: int = 10):
        self.n_clusters  = n_clusters
        self.method      = method
        self.pca_dim     = pca_dim
        self.scaler      = StandardScaler()
        self.pca         = PCA(n_components=pca_dim, random_state=42)
        self.cluster_model = None
        self.feature_cols: list[str] = []
        self.well_labels: dict = {}  # well_id → cluster_id

    def fit(self, summary_df: pd.DataFrame) -> "WellClusterer":
        """
        Fit clustering on well summary features.
        Automatically selects best K using silhouette score if n_clusters='auto'.
        """
        X = summary_df.select_dtypes(include=[np.number]).fillna(0)
        self.feature_cols = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        n_dim = min(self.pca_dim, X_scaled.shape[1], X_scaled.shape[0] - 1)
        n_dim = max(n_dim, 1)
        self.pca = PCA(n_components=n_dim, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        # Auto-select K
        if self.n_clusters == 0:
            self.n_clusters = self._auto_select_k(X_pca)

        if self.method == "kmeans":
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters, random_state=42, n_init=10
            )
        else:
            self.cluster_model = AgglomerativeClustering(n_clusters=self.n_clusters)

        labels = self.cluster_model.fit_predict(X_pca)

        for well_id, label in zip(summary_df.index, labels):
            self.well_labels[well_id] = int(label)

        # Log cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"[clustering] {self.n_clusters} clusters fitted:")
        for c, n in zip(unique, counts):
            print(f"  Cluster {c}: {n} wells")

        return self

    def predict(self, summary_df: pd.DataFrame) -> np.ndarray:
        """Assign cluster labels to new wells."""
        X = summary_df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        n_dim = self.pca.n_components_
        X_pca = self.pca.transform(X_scaled)[:, :n_dim]
        return self.cluster_model.predict(X_pca)

    def assign_to_df(
        self,
        df: pd.DataFrame,
        summary_df: pd.DataFrame,
        well_col: str = "well_id",
    ) -> pd.DataFrame:
        """Add cluster_id and cluster distance features to per-row dataframe."""
        out = df.copy()
        out["cluster_id"] = out[well_col].map(self.well_labels).fillna(-1).astype(int)

        # Cluster distance: distance to own cluster center (confidence proxy)
        X = summary_df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        if hasattr(self.cluster_model, "cluster_centers_"):
            centers = self.cluster_model.cluster_centers_
            labels = np.array(list(self.well_labels.values()))
            dists = {}
            for well_id, (x_row, lbl) in zip(summary_df.index, zip(X_pca, labels)):
                center = centers[lbl]
                dists[well_id] = float(np.linalg.norm(x_row - center))
            out["cluster_dist"] = out[well_col].map(dists).fillna(999.0)
        else:
            out["cluster_dist"] = 0.0

        return out

    def _auto_select_k(self, X: np.ndarray, k_range: range = range(3, 10)) -> int:
        """Select K with best silhouette score."""
        best_k, best_score = 5, -1.0
        for k in k_range:
            if k >= len(X):
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = km.fit_predict(X)
            try:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score, best_k = score, k
            except Exception:
                pass
        print(f"[clustering] Auto-selected K={best_k} (silhouette={best_score:.3f})")
        return best_k

    def get_cluster_wells(self, cluster_id: int) -> list:
        """Return well IDs belonging to a cluster."""
        return [wid for wid, cid in self.well_labels.items() if cid == cluster_id]

    def get_similar_wells(self, well_id, k: int = 5) -> list[tuple]:
        """
        Return K most similar wells by embedding distance.
        Returns: [(well_id, distance), ...]
        """
        # This would need access to summary_df - keep for now as placeholder
        same_cluster = self.get_cluster_wells(self.well_labels.get(well_id, -1))
        return [(w, 0.0) for w in same_cluster if w != well_id][:k]


def train_per_cluster_models(
    df: pd.DataFrame,
    clusterer: WellClusterer,
    feature_cols: list[str],
    target_col:   str = "TVT",
    well_col:     str = "well_id",
) -> dict:
    """
    Train one CatBoost model per cluster.
    Returns {cluster_id: model}.
    """
    try:
        from catboost import CatBoostRegressor
        ModelClass = lambda: CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=7, verbose=0, random_seed=42
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        ModelClass = lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
        )

    models = {}
    for cluster_id in range(clusterer.n_clusters):
        wells_in_cluster = clusterer.get_cluster_wells(cluster_id)
        cluster_data = df[df[well_col].isin(wells_in_cluster)]

        mask = cluster_data[target_col].notna()
        if mask.sum() < 50:
            print(f"[clustering] Cluster {cluster_id}: too few samples ({mask.sum()}), skipping")
            continue

        avail_cols = [c for c in feature_cols if c in cluster_data.columns]
        X = cluster_data[mask][avail_cols].select_dtypes(include=[np.number]).fillna(0)
        y = cluster_data[mask][target_col].values

        model = ModelClass()
        model.fit(X, y)
        models[cluster_id] = model
        print(f"[clustering] Cluster {cluster_id}: trained on {len(y):,} samples from {len(wells_in_cluster)} wells")

    return models
