# =============================================================
# src/evaluation.py – Model Evaluation & Metrics
# =============================================================
# Responsibility: Compute classification metrics, print a
# comparison table, and generate confusion matrices.
# =============================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class Evaluator:
    """
    Computes and stores evaluation metrics for multiple models
    so they can be compared side-by-side.
    """

    def __init__(self):
        # {model_name: {metric_name: value}}
        self.results: dict = {}
        self.confusion_matrices: dict = {}

    # ----------------------------------------------------------
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> dict:
        """
        Compute metrics for a single model and store internally.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels (1 = normal, 0 = anomaly).
        y_pred : np.ndarray
            Predicted labels.
        model_name : str
            Human-readable name for console output.

        Returns
        -------
        dict
            Dictionary of metric name → value.
        """
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }
        self.results[model_name] = metrics
        self.confusion_matrices[model_name] = cm

        return metrics

    # ----------------------------------------------------------
    def print_report(self):
        """Print a formatted comparison table for all evaluated models."""
        if not self.results:
            print("[Evaluator] No models evaluated yet.")
            return

        print("\n" + "=" * 72)
        print("MODEL EVALUATION REPORT")
        print("=" * 72)
        header = f"{'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}"
        print(header)
        print("-" * 72)

        for name, m in self.results.items():
            print(
                f"{name:<25s} "
                f"{m['accuracy']:>10.4f} "
                f"{m['precision']:>10.4f} "
                f"{m['recall']:>10.4f} "
                f"{m['f1_score']:>10.4f}"
            )

        print("-" * 72)

        # Highlight best model
        best_model = max(self.results, key=lambda k: self.results[k]["f1_score"])
        print(f"\n* Best model by F1-Score: {best_model} "
              f"(F1 = {self.results[best_model]['f1_score']:.4f})")

        # Print confusion matrices
        for name, cm in self.confusion_matrices.items():
            print(f"\n[Confusion Matrix - {name}]")
            print(f"  Predicted ->  Normal   Anomaly")
            print(f"  Actual (v)")
            print(f"  Normal     {cm[1][1]:>8d}  {cm[1][0]:>8d}")
            print(f"  Anomaly    {cm[0][1]:>8d}  {cm[0][0]:>8d}")

        print()
