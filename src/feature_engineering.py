# =============================================================
# src/feature_engineering.py – Feature Engineering Utilities
# =============================================================
# Responsibility: Create derived features that capture network
# traffic behaviour more effectively for anomaly detection.
# =============================================================

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Extracts / creates additional features from the raw dataset
    that help ML models distinguish normal from anomalous traffic.

    Key engineered features:
      • byte_ratio         – ratio of src_bytes to dst_bytes
      • total_bytes        – sum of src and dst bytes
      • error_rate_sum     – combined error rate signal
      • srv_ratio          – ratio of srv_count to count (connection density)
      • host_srv_ratio     – ratio of dst_host_srv_count to dst_host_count
      • is_heavy_traffic   – boolean flag for high byte count
      • log_src_bytes      – log-transform to reduce skewness
      • log_dst_bytes      – log-transform to reduce skewness
    """

    @staticmethod
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features IN PLACE and return the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains the raw NSL-KDD columns.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with new columns appended.
        """
        print("[FeatureEngineer] Creating derived features ...")

        # ----------------------------------------------------------
        # 1. Byte-level features
        # ----------------------------------------------------------
        # Total bytes transferred in the connection
        df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]

        # Ratio of sent-to-received bytes (asymmetry signal)
        df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)

        # Log-transform to compress heavy tail
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        df["log_dst_bytes"] = np.log1p(df["dst_bytes"])

        # ----------------------------------------------------------
        # 2. Flag for heavy traffic (> 10 000 total bytes)
        # ----------------------------------------------------------
        df["is_heavy_traffic"] = (df["total_bytes"] > 10_000).astype(int)

        # ----------------------------------------------------------
        # 3. Error-rate aggregation
        # ----------------------------------------------------------
        # Combining multiple error signals into one metric
        df["error_rate_sum"] = (
            df["serror_rate"]
            + df["srv_serror_rate"]
            + df["rerror_rate"]
            + df["srv_rerror_rate"]
        )

        # ----------------------------------------------------------
        # 4. Connection-density ratio
        # ----------------------------------------------------------
        df["srv_ratio"] = df["srv_count"] / (df["count"] + 1)

        # ----------------------------------------------------------
        # 5. Host-level service ratio
        # ----------------------------------------------------------
        df["host_srv_ratio"] = df["dst_host_srv_count"] / (
            df["dst_host_count"] + 1
        )

        new_features = [
            "total_bytes", "byte_ratio", "log_src_bytes", "log_dst_bytes",
            "is_heavy_traffic", "error_rate_sum", "srv_ratio", "host_srv_ratio",
        ]
        print(f"[FeatureEngineer] Added {len(new_features)} new features:")
        for f in new_features:
            print(f"  - {f}")
        print()

        return df
