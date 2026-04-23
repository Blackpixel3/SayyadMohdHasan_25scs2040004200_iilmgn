# =============================================================
# src/models/kmeans_model.py – K-Means Clustering Detector
# =============================================================
# K-Means groups data points into 'k' clusters.  For anomaly
# detection we use k=2 (normal / anomaly) and then label the
# cluster whose centroid is farther from the origin as the
# "anomaly" cluster.  We also compute each point's distance
# to its nearest centroid — a useful anomaly proximity score.
# =============================================================

import numpy as np
from sklearn.cluster import KMeans

from config import KMEANS_N_CLUSTERS, KMEANS_N_INIT, KMEANS_RANDOM_STATE


class KMeansDetector:
    """
    Wrapper around sklearn's KMeans that provides:
      • Training
      • Prediction (binary: 1 = normal, 0 = anomaly)
      • Distance-to-centroid scores (for hybrid fusion)
    """

    def __init__(
        self,
        n_clusters: int = KMEANS_N_CLUSTERS,
        n_init: int = KMEANS_N_INIT,
        random_state: int = KMEANS_RANDOM_STATE,
    ):
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )
        self.name = "K-Means"
        self.anomaly_cluster: int = -1  # set after fitting
        print(f"[{self.name}] Initialised (k={n_clusters})")

    # ----------------------------------------------------------
    def fit(self, X_train: np.ndarray):
        """Fit K-Means and identify which cluster is 'anomaly'."""
        print(f"[{self.name}] Clustering {X_train.shape[0]} samples ...")
        self.model.fit(X_train)

        # Heuristic: the cluster whose centroid has a larger L2 norm
        # from the origin is more likely to be the anomaly cluster
        # (since normal traffic clumps near the centre after scaling).
        norms = np.linalg.norm(self.model.cluster_centers_, axis=1)
        self.anomaly_cluster = int(np.argmax(norms))
        print(f"[{self.name}] Training complete [OK]  "
              f"(anomaly cluster = {self.anomaly_cluster})")

    # ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary predictions: 1 = normal, 0 = anomaly.
        Points assigned to the anomaly cluster → 0.
        """
        cluster_labels = self.model.predict(X)
        return (cluster_labels != self.anomaly_cluster).astype(int)

    # ----------------------------------------------------------
    def centroid_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute each sample's distance to its NEAREST centroid.
        Higher distance → more anomaly-like.
        """
        centroids = self.model.cluster_centers_
        # X shape: (n_samples, n_features)
        # centroids shape: (k, n_features)
        # distances shape: (n_samples, k)
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        return np.min(dists, axis=1)
