# =============================================================
# src/models/isolation_forest_model.py – Isolation Forest Detector
# =============================================================
# Isolation Forest is an unsupervised anomaly detection algorithm.
# It isolates anomalies by randomly selecting a feature and
# then randomly selecting a split value. Anomalies require
# fewer splits → shorter path length → lower anomaly score.
# =============================================================

import numpy as np
from sklearn.ensemble import IsolationForest

from config import ISO_CONTAMINATION, ISO_N_ESTIMATORS, ISO_RANDOM_STATE


class IsolationForestDetector:
    """
    Wrapper around sklearn's IsolationForest that provides:
      • Training
      • Prediction (binary: 1 = normal, 0 = anomaly)
      • Raw anomaly scores (for hybrid fusion)
    """

    def __init__(
        self,
        contamination: float = ISO_CONTAMINATION,
        n_estimators: int = ISO_N_ESTIMATORS,
        random_state: int = ISO_RANDOM_STATE,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,  # use all CPU cores
        )
        self.name = "Isolation Forest"
        print(f"[{self.name}] Initialised (contamination={contamination}, "
              f"n_estimators={n_estimators})")

    # ----------------------------------------------------------
    def fit(self, X_train: np.ndarray):
        """Fit the Isolation Forest on training data."""
        print(f"[{self.name}] Training on {X_train.shape[0]} samples ...")
        self.model.fit(X_train)
        print(f"[{self.name}] Training complete [OK]")

    # ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary predictions: 1 = normal, 0 = anomaly.
        (sklearn returns +1 for inliers and -1 for outliers.)
        """
        raw = self.model.predict(X)
        return (raw == 1).astype(int)

    # ----------------------------------------------------------
    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return the raw anomaly decision scores.
        Lower (more negative) scores indicate stronger anomalies.
        """
        return self.model.decision_function(X)
