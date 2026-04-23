# =============================================================
# main.py – Master Pipeline
# =============================================================
# AI-Based Behavioral Deep Packet Inspection (DPI) for
# Anomaly Detection – Final Year Project
#
# This script orchestrates the entire pipeline:
#   1. Load & clean the NSL-KDD dataset
#   2. Engineer behavioural features
#   3. Preprocess (encode, split, scale)
#   4. Train Isolation Forest & K-Means models
#   5. Compute hybrid anomaly scores (innovation)
#   6. Evaluate all three detection strategies
#   7. Generate visualizations
#   8. Run a real-time simulation
#
# Usage:
#   python main.py
#
# Dataset:
#   Place KDDTrain+.txt inside the data/ folder.
#   Download from: https://www.unb.ca/cic/datasets/nsl.html
# =============================================================

import os
import sys
import io
import warnings
import joblib

# --- Force UTF-8 stdout on Windows to avoid encoding errors ---
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Suppress sklearn / numpy warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Add project root to path so imports work from anywhere ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, MODEL_DIR

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessor import Preprocessor
from src.models.isolation_forest_model import IsolationForestDetector
from src.models.kmeans_model import KMeansDetector
from src.models.hybrid_detector import HybridAnomalyDetector
from src.evaluation import Evaluator
from src.visualization import Visualizer
from src.realtime_simulator import RealTimeSimulator


def print_banner():
    """Print a project banner for console output."""
    banner = """
+==============================================================+
|   AI-Based Behavioral Deep Packet Inspection (DPI)           |
|   for Anomaly Detection                                      |
|                                                              |
|   Models : Isolation Forest  x  K-Means  x  Hybrid Fusion   |
|   Dataset: NSL-KDD                                          |
+==============================================================+
    """
    print(banner)


def main():
    print_banner()

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 1 – Load the dataset                           ║
    # ╚═══════════════════════════════════════════════════════╝
    print("\n" + "=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)
    loader = DataLoader()
    df = loader.load()

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 2 – Feature Engineering                        ║
    # ╚═══════════════════════════════════════════════════════╝
    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)
    df = FeatureEngineer.engineer(df)

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 3 – Preprocessing                              ║
    # ╚═══════════════════════════════════════════════════════╝
    print("=" * 60)
    print("STEP 3: PREPROCESSING")
    print("=" * 60)
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 4 – Train Models                               ║
    # ╚═══════════════════════════════════════════════════════╝
    print("=" * 60)
    print("STEP 4: MODEL TRAINING")
    print("=" * 60)

    # --- 4a. Isolation Forest ---
    iso_detector = IsolationForestDetector()
    iso_detector.fit(X_train)

    # --- 4b. K-Means ---
    kmeans_detector = KMeansDetector()
    kmeans_detector.fit(X_train)

    # --- 4c. Hybrid Detector ---
    hybrid_detector = HybridAnomalyDetector()
    
    # Fit the hybrid detector with normalisation bounds from training data
    iso_scores_train = iso_detector.decision_scores(X_train)
    km_distances_train = kmeans_detector.centroid_distances(X_train)
    hybrid_detector.fit(iso_scores_train, km_distances_train)

    # Get raw scores on test set
    iso_scores_test = iso_detector.decision_scores(X_test)
    km_distances_test = kmeans_detector.centroid_distances(X_test)

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 5 – Predictions                                ║
    # ╚═══════════════════════════════════════════════════════╝
    print("\n" + "=" * 60)
    print("STEP 5: PREDICTIONS")
    print("=" * 60)

    iso_preds = iso_detector.predict(X_test)
    kmeans_preds = kmeans_detector.predict(X_test)
    hybrid_preds, hybrid_scores, hybrid_threshold = hybrid_detector.predict(
        iso_scores_test, km_distances_test, use_adaptive=True
    )

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 6 – Evaluation                                 ║
    # ╚═══════════════════════════════════════════════════════╝
    print("\n" + "=" * 60)
    print("STEP 6: EVALUATION")
    print("=" * 60)

    evaluator = Evaluator()
    evaluator.evaluate(y_test, iso_preds, "Isolation Forest")
    evaluator.evaluate(y_test, kmeans_preds, "K-Means")
    evaluator.evaluate(y_test, hybrid_preds, "Hybrid (IF + KM)")
    evaluator.print_report()

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 7 – Visualization                              ║
    # ╚═══════════════════════════════════════════════════════╝
    print("=" * 60)
    print("STEP 7: VISUALIZATION")
    print("=" * 60)

    viz = Visualizer()
    viz.plot_class_distribution(y_test)
    viz.plot_score_distributions(
        iso_scores_test, km_distances_test, hybrid_scores,
        y_test, hybrid_threshold,
    )
    viz.plot_anomaly_scatter(iso_scores_test, km_distances_test, y_test)
    viz.plot_confusion_matrices(evaluator.confusion_matrices)
    viz.plot_model_comparison(evaluator.results)

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 8 – Real-Time Simulation                       ║
    # ╚═══════════════════════════════════════════════════════╝
    print("=" * 60)
    print("STEP 8: REAL-TIME SIMULATION")
    print("=" * 60)

    simulator = RealTimeSimulator(
        iso_model=iso_detector,
        kmeans_model=kmeans_detector,
        hybrid_detector=hybrid_detector,
        scaler=preprocessor.scaler,
        feature_names=preprocessor.feature_names,
    )
    sim_results = simulator.run(X_test, y_test)

    # ╔═══════════════════════════════════════════════════════╗
    # ║  STEP 9 – Save Models                                ║
    # ╚═══════════════════════════════════════════════════════╝
    print("=" * 60)
    print("STEP 9: SAVING TRAINED MODELS")
    print("=" * 60)

    joblib.dump(iso_detector, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(kmeans_detector, os.path.join(MODEL_DIR, "kmeans.pkl"))
    joblib.dump(hybrid_detector, os.path.join(MODEL_DIR, "hybrid_detector.pkl"))
    joblib.dump(preprocessor.scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    # Save a sample of unmodified test data so the dashboard can stream realistic packets
    X_test_unscaled = preprocessor.scaler.inverse_transform(X_test)
    test_data = {
        "X_test_unscaled": X_test_unscaled[:2000],
        "y_test": y_test[:2000]
    }
    joblib.dump(test_data, os.path.join(MODEL_DIR, "test_data.pkl"))
    print(f"[main] Models and test data sample saved to {MODEL_DIR}/")

    # ╔═══════════════════════════════════════════════════════╗
    # ║  DONE                                                 ║
    # ╚═══════════════════════════════════════════════════════╝
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print("=" * 60)
    print(f"  Visualizations → {OUTPUT_DIR}/")
    print(f"  Saved models   → {MODEL_DIR}/")
    print(f"  Flask dashboard → python app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
