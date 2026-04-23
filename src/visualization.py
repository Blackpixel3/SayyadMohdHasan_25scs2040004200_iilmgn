# =============================================================
# src/visualization.py – Charts & Plots
# =============================================================
# Responsibility: Generate publication-quality visualizations
# that compare model performance and show anomaly distributions.
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR


class Visualizer:
    """
    Generates and saves six key plots:
      1. Isolation Forest anomaly score distribution
      2. K-Means centroid distance distribution
      3. Hybrid anomaly score scatter
      4. Confusion matrix heatmaps
      5. Model comparison bar chart
      6. Anomaly vs normal traffic class distribution
    """

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Use a clean style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")
        print(f"[Visualizer] Output directory: {OUTPUT_DIR}")

    # ----------------------------------------------------------
    # 1. Score distribution
    # ----------------------------------------------------------
    def plot_score_distributions(
        self,
        iso_scores: np.ndarray,
        km_distances: np.ndarray,
        hybrid_scores: np.ndarray,
        y_true: np.ndarray,
        threshold: float,
    ):
        """Three-panel figure: IF scores, KM distances, Hybrid scores."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        normal_mask = y_true == 1
        anomaly_mask = y_true == 0

        # Panel 1 – Isolation Forest
        axes[0].hist(iso_scores[normal_mask], bins=60, alpha=0.6,
                     label="Normal", color="#2ecc71")
        axes[0].hist(iso_scores[anomaly_mask], bins=60, alpha=0.6,
                     label="Anomaly", color="#e74c3c")
        axes[0].set_title("Isolation Forest – Decision Scores", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Score")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        # Panel 2 – K-Means
        axes[1].hist(km_distances[normal_mask], bins=60, alpha=0.6,
                     label="Normal", color="#2ecc71")
        axes[1].hist(km_distances[anomaly_mask], bins=60, alpha=0.6,
                     label="Anomaly", color="#e74c3c")
        axes[1].set_title("K-Means – Centroid Distances", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Distance")
        axes[1].set_ylabel("Count")
        axes[1].legend()

        # Panel 3 – Hybrid
        axes[2].hist(hybrid_scores[normal_mask], bins=60, alpha=0.6,
                     label="Normal", color="#2ecc71")
        axes[2].hist(hybrid_scores[anomaly_mask], bins=60, alpha=0.6,
                     label="Anomaly", color="#e74c3c")
        axes[2].axvline(threshold, color="#f39c12", linestyle="--", linewidth=2,
                        label=f"Threshold = {threshold:.3f}")
        axes[2].set_title("Hybrid – Fused Anomaly Scores", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Hybrid Score")
        axes[2].set_ylabel("Count")
        axes[2].legend()

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "score_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved: {path}")

    # ----------------------------------------------------------
    # 2. Anomaly scatter plot
    # ----------------------------------------------------------
    def plot_anomaly_scatter(
        self,
        iso_scores: np.ndarray,
        km_distances: np.ndarray,
        y_true: np.ndarray,
    ):
        """2D scatter: IF score vs KM distance, coloured by true label."""
        fig, ax = plt.subplots(figsize=(10, 7))

        scatter = ax.scatter(
            iso_scores, km_distances,
            c=y_true, cmap="RdYlGn", alpha=0.4, s=8, edgecolors="none",
        )
        ax.set_xlabel("Isolation Forest Score", fontsize=12)
        ax.set_ylabel("K-Means Centroid Distance", fontsize=12)
        ax.set_title("Anomaly Landscape  (Green = Normal, Red = Anomaly)",
                      fontsize=14, fontweight="bold")
        plt.colorbar(scatter, label="Label (1=Normal, 0=Anomaly)")
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, "anomaly_scatter.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved: {path}")

    # ----------------------------------------------------------
    # 3. Confusion matrices
    # ----------------------------------------------------------
    def plot_confusion_matrices(self, confusion_dict: dict):
        """Side-by-side confusion matrix heatmaps."""
        n_models = len(confusion_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for ax, (name, cm) in zip(axes, confusion_dict.items()):
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Anomaly (0)", "Normal (1)"],
                yticklabels=["Anomaly (0)", "Normal (1)"],
                ax=ax,
            )
            ax.set_title(f"{name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved: {path}")

    # ----------------------------------------------------------
    # 4. Model comparison bar chart
    # ----------------------------------------------------------
    def plot_model_comparison(self, results: dict):
        """Grouped bar chart comparing metrics across models."""
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        model_names = list(results.keys())
        n_models = len(model_names)
        x = np.arange(len(metrics))
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

        for i, model in enumerate(model_names):
            values = [results[model][m] for m in metrics]
            bars = ax.bar(x + i * width, values, width,
                          label=model, color=colors[i % len(colors)],
                          edgecolor="white", linewidth=0.5)
            # Add value labels on top
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved: {path}")

    # ----------------------------------------------------------
    # 5. Class distribution pie chart
    # ----------------------------------------------------------
    def plot_class_distribution(self, y_true: np.ndarray):
        """Pie chart of normal vs anomaly traffic."""
        normal_count = int(np.sum(y_true == 1))
        anomaly_count = int(np.sum(y_true == 0))

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            [normal_count, anomaly_count],
            labels=["Normal Traffic", "Anomalous Traffic"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
            startangle=90,
            explode=(0, 0.05),
            shadow=True,
        )
        for t in autotexts:
            t.set_fontsize(12)
            t.set_fontweight("bold")
        ax.set_title("Dataset Class Distribution", fontsize=14, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "class_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved: {path}")
