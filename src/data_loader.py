# =============================================================
# src/data_loader.py – Dataset Loading & Initial Cleaning
# =============================================================
# Responsibility: Load the raw NSL-KDD CSV, validate columns,
# handle missing values, and return a clean DataFrame.
# =============================================================

import os
import pandas as pd
import numpy as np

from config import (
    TRAIN_DATA_FILE,
    NSL_KDD_COLUMNS,
)


class DataLoader:
    """
    Loads the NSL-KDD dataset from disk and performs initial
    data-quality checks (missing values, corrupt rows, etc.).
    """

    def __init__(self, filepath: str = None):
        """
        Parameters
        ----------
        filepath : str, optional
            Path to the dataset CSV. Defaults to config.TRAIN_DATA_FILE.
        """
        self.filepath = filepath or TRAIN_DATA_FILE

    # ----------------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """
        Load the raw dataset and return a clean DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with all original columns plus a
            binary label column (1 = normal, 0 = anomaly).
        """
        self._check_file_exists()

        print("[DataLoader] Loading dataset from:", self.filepath)
        df = pd.read_csv(
            self.filepath,
            header=None,
            names=NSL_KDD_COLUMNS,
            sep=",",
            engine="python",
            on_bad_lines="skip",  # skip malformed rows gracefully
        )

        # Drop the 'difficulty' column (not needed for modelling)
        if "difficulty" in df.columns:
            df = df.drop(columns=["difficulty"])

        df = df.reset_index(drop=True)

        # Create a binary label: 1 = normal, 0 = attack/anomaly
        df["binary_label"] = (df["label"] == "normal").astype(int)

        # Handle missing / infinite values
        df = self._handle_missing(df)

        print(f"[DataLoader] Loaded {len(df)} rows x {df.shape[1]} columns")
        print("[DataLoader] Class distribution:")
        print(df["binary_label"].value_counts().to_string())
        print()

        return df

    # ----------------------------------------------------------
    # PRIVATE HELPERS
    # ----------------------------------------------------------
    def _check_file_exists(self):
        """Raise a helpful error if the dataset file is missing."""
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f" Dataset not found!\n"
                f" Expected: {self.filepath}\n\n"
                f" Please download the NSL-KDD dataset and place\n"
                f" 'KDDTrain+.txt' inside the data/ folder.\n"
                f" Download: https://www.unb.ca/cic/datasets/nsl.html\n"
                f"{'='*60}"
            )

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill numeric NaNs with column medians and categorical
        NaNs with the mode (most frequent value).
        """
        print("[DataLoader] Handling missing values ...")

        # Numeric columns → fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if df[numeric_cols].isnull().any().any():
            df[numeric_cols] = df[numeric_cols].fillna(
                df[numeric_cols].median()
            )
            print("  -> Filled numeric NaNs with column medians")
        else:
            print("  -> No numeric NaN values found")

        # Categorical columns → fill with mode
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  -> Filled '{col}' NaNs with mode: {mode_val}")

        # Replace infinities with NaN, then fill again
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        return df
