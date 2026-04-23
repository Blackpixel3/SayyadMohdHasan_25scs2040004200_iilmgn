# =============================================================
# src/models/hybrid_detector.py – Hybrid Anomaly Scorer
# =============================================================
# INNOVATION COMPONENT
# ---------------------
# This module implements a *hybrid anomaly score* that fuses
# the outputs of two independent detection strategies:
#
#   1. Isolation Forest  → anomaly decision score
#   2. K-Means           → distance-to-nearest-centroid
#
# By combining both signals with a tuneable weight (alpha), we
# get a more robust detector that benefits from the strengths
# of each approach:
#   • Isolation Forest excels at detecting point anomalies
#     that deviate from the general data distribution.
#   • K-Means captures cluster-structure anomalies where a
#     point falls between or far from any cluster centre.
#
# Additionally, we implement DYNAMIC THRESHOLD ADAPTATION that
# adjusts the detection threshold based on the score distribution,
# rather than relying on a single fixed cut-off.
# =============================================================

import numpy as np
from config import HYBRID_ALPHA, HYBRID_THRESHOLD


class HybridAnomalyDetector:
    """
    Fuses Isolation Forest and K-Means anomaly signals into
    a single hybrid score, then applies an adaptive threshold.

    Score semantics:
        0.0 → definitely normal
        1.0 → definitely anomaly
    """

    def __init__(
        self,
        alpha: float = HYBRID_ALPHA,
        base_threshold: float = HYBRID_THRESHOLD,
    ):
        """
        Parameters
        ----------
        alpha : float
            Weight of the Isolation Forest component (0–1).
            The K-Means component receives weight (1 - alpha).
        base_threshold : float
            Base detection threshold. The adaptive algorithm may
            adjust this per batch.
        """
        self.alpha = alpha
        self.base_threshold = base_threshold
        self.adaptive_threshold = base_threshold
        self.name = "Hybrid Detector"
        
        self.iso_min = None
        self.iso_max = None
        self.km_min = None
        self.km_max = None

        print(f"[{self.name}] Initialised (alpha={alpha}, "
              f"base_threshold={base_threshold})")

    # ----------------------------------------------------------
    # FIT: Store min/max bounds for normalisation based on train/test data
    # ----------------------------------------------------------
    def fit(self, iso_scores: np.ndarray, kmeans_distances: np.ndarray):
        """Record the min and max values for normalising single points later."""
        self.iso_min, self.iso_max = iso_scores.min(), iso_scores.max()
        self.km_min, self.km_max = kmeans_distances.min(), kmeans_distances.max()
        print(f"[{self.name}] Fitted normalisation bounds.")

    # ----------------------------------------------------------
    # CORE: Compute hybrid scores
    # ----------------------------------------------------------
    def compute_scores(
        self,
        iso_scores: np.ndarray,
        kmeans_distances: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the fused hybrid anomaly score.

        Parameters
        ----------
        iso_scores : np.ndarray
            Raw Isolation Forest decision scores (lower = more anomalous).
        kmeans_distances : np.ndarray
            Distance of each sample to its nearest K-Means centroid
            (higher = more anomalous).

        Returns
        -------
        np.ndarray
            Hybrid scores in [0, 1] where 1 = most anomalous.
        """
        # If not fitted yet, fallback to local batch min/max
        iso_min = self.iso_min if self.iso_min is not None else iso_scores.min()
        iso_max = self.iso_max if self.iso_max is not None else iso_scores.max()
        km_min = self.km_min if self.km_min is not None else kmeans_distances.min()
        km_max = self.km_max if self.km_max is not None else kmeans_distances.max()

        # --- Normalise Isolation Forest scores to [0, 1] ---
        iso_norm = (iso_scores - iso_min) / ((iso_max - iso_min) + 1e-8)
        iso_norm = np.clip(iso_norm, 0, 1)
        # Invert: in sklearn, higher score = more normal
        iso_anomaly = 1.0 - iso_norm

        # --- Normalise K-Means distances to [0, 1] ---
        km_anomaly = (kmeans_distances - km_min) / ((km_max - km_min) + 1e-8)
        km_anomaly = np.clip(km_anomaly, 0, 1)

        # --- Weighted fusion ---
        hybrid = self.alpha * iso_anomaly + (1.0 - self.alpha) * km_anomaly

        return hybrid

    # ----------------------------------------------------------
    # INNOVATION: Adaptive threshold
    # ----------------------------------------------------------
    def adapt_threshold(self, scores: np.ndarray) -> float:
        """
        Dynamically adapt the threshold using the score distribution.

        Strategy: set the threshold at (mean + 1.5 × std) of the
        hybrid scores.  This adapts to the data rather than using
        a fixed value, making the detector more robust across
        different traffic patterns.

        Parameters
        ----------
        scores : np.ndarray
            Hybrid anomaly scores for a batch.

        Returns
        -------
        float
            The adapted threshold.
        """
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        self.adaptive_threshold = float(
            np.clip(mean_score + 1.5 * std_score, 0.1, 0.95)
        )
        print(f"[{self.name}] Adaptive threshold: {self.adaptive_threshold:.4f} "
              f"(mean={mean_score:.4f}, std={std_score:.4f})")
        return self.adaptive_threshold

    # ----------------------------------------------------------
    # PREDICT
    # ----------------------------------------------------------
    def predict(
        self,
        iso_scores: np.ndarray,
        kmeans_distances: np.ndarray,
        use_adaptive: bool = True,
    ) -> tuple:
        """
        Full hybrid prediction pipeline.

        Returns
        -------
        predictions : np.ndarray   (1 = normal, 0 = anomaly)
        hybrid_scores : np.ndarray (continuous 0–1 score)
        threshold : float          (threshold that was used)
        """
        hybrid_scores = self.compute_scores(iso_scores, kmeans_distances)

        if use_adaptive:
            threshold = self.adapt_threshold(hybrid_scores)
        else:
            threshold = self.base_threshold

        # Scores ABOVE threshold → anomaly (0), below → normal (1)
        predictions = (hybrid_scores < threshold).astype(int)

        return predictions, hybrid_scores, threshold
