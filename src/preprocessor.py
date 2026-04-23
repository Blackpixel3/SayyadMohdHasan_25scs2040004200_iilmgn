# =============================================================
# src/preprocessor.py – Feature Encoding & Scaling
# =============================================================
# Responsibility: Encode categoricals, select behavioural
# features, and apply StandardScaler normalisation.
# =============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from config import (
    CATEGORICAL_COLS,
    BEHAVIORAL_FEATURES,
    TEST_SIZE,
    SPLIT_RANDOM_STATE,
)


class Preprocessor:
    """
    End-to-end preprocessing pipeline:
      1. Encode categorical features with LabelEncoder
      2. Build the feature matrix (behavioural + encoded cols)
      3. Train/test split
      4. StandardScaler normalisation
    """

    def __init__(self):
        self.label_encoders: dict = {}   # col_name → fitted LabelEncoder
        self.scaler = StandardScaler()
        self.feature_names: list = []

    # ----------------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame):
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Clean DataFrame from DataLoader (must contain
            'binary_label' column).

        Returns
        -------
        X_train_scaled, X_test_scaled : np.ndarray
        y_train, y_test : np.ndarray
        """
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)

        # Step 1 – Encode categoricals
        df = self._encode_categoricals(df)

        # Step 2 – Build the feature matrix
        X, y = self._build_feature_matrix(df)

        # Step 3 – Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify=y,
        )
        print(f"\n[Preprocessor] Train/Test split ({1-TEST_SIZE:.0%} / {TEST_SIZE:.0%}):")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test : {X_test.shape[0]} samples")

        # Step 4 – Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("[Preprocessor] Features scaled with StandardScaler\n")

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values

    # ----------------------------------------------------------
    # PRIVATE HELPERS
    # ----------------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode each categorical column and store encoders."""
        print("\n[Preprocessor] Encoding categorical features ...")
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            encoded_col = col + "_enc"
            df[encoded_col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  {col:20s} -> {encoded_col} ({len(le.classes_)} unique values)")
        return df

    def _build_feature_matrix(self, df: pd.DataFrame):
        """Select behavioural + encoded columns for modelling."""
        encoded_cols = [c + "_enc" for c in CATEGORICAL_COLS]
        self.feature_names = BEHAVIORAL_FEATURES + encoded_cols

        print(f"\n[Preprocessor] Selected {len(self.feature_names)} features:")
        for i, name in enumerate(self.feature_names, 1):
            print(f"  {i:2d}. {name}")

        X = df[self.feature_names].copy()
        y = df["binary_label"].copy()  # 1 = normal, 0 = anomaly
        return X, y
