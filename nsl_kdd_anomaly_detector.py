# File: nsl_kdd_anomaly_detector.py
# AI‑Based Behavioral Deep Packet Inspection (DPI) for Anomaly Detection
# Final‑year project – Senior Cybersecurity Engineer + ML Researcher style

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DATASET_PATH = "data/KDDTrain+.txt"  # Replace with your train file
COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Critical behavioral features (packet‑level + protocol)
FEATURE_COLS = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# --- DATA LOADING & PREPROCESSING ---
def load_and_preprocess():
    """Load NSL‑KDD train data and do preprocessing."""
    print("[1/7] Loading dataset from:", DATASET_PATH)
    df = pd.read_csv(
        DATASET_PATH,
        header=None,
        names=COLUMNS,
        sep=",",
        engine="python",
        on_bad_lines="skip"  # ignore malformed lines
    )

    # Remove difficulty (last column) and reset index
    df = df.drop(columns=["difficulty"]).reset_index(drop=True)

    # Convert normal/attack to binary label
    # NSL‑KDD: "normal" vs various attacks
    df["binary_label"] = (df["label"] == "normal").astype(int)  # 1 = normal, 0 = anomaly

    print("Raw data shape:", df.shape)
    print("Normal (1) vs Anomaly (0):")
    print(df["binary_label"].value_counts())

    # --- Handle missing values ---
    print("[2/7] Handling missing values...")
    # NSL‑KDD should have very few NaNs; fill numeric with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    # --- Encode categorical features ---
    print("[3/7] Encoding categorical features...")
    # Categorical features in NSL‑KDD
    cat_cols = ["protocol_type", "service", "flag"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        le_dict[col] = le
        print(f"  {col} -> {col}_enc with {len(le.classes_)} classes")

    # --- Combine: numeric + encoded categorical into feature matrix ---
    feature_cols = FEATURE_COLS + [c + "_enc" for c in cat_cols]
    X = df[feature_cols].copy()
    y = df["binary_label"].copy()  # 1 = normal, 0 = attack

    print("Feature matrix shape:", X.shape)
    return X, y, le_dict


# --- SCALING ---
def scale_features(X_train, X_test):
    """Scale numeric features using StandardScaler."""
    print("[4/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Shape after scaling:", X_train_scaled.shape, X_test_scaled.shape)
    return X_train_scaled, X_test_scaled, scaler


# --- MODEL TRAINING ---
def train_models(X_train, y_train, X_test, y_test):
    """Train Isolation Forest and K‑Means; return scores and predictions."""
    print("[5/7] Training models...")

    # --- Isolation Forest (Anomaly Detection) ---
    print("  [a] Isolation Forest (contamination=0.1)...")
    # We assume roughly 10% anomalies; tune for your runs
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train)

    iso_scores_train = iso_forest.decision_function(X_train)
    iso_scores_test = iso_forest.decision_function(X_test)

    iso_pred_train = iso_forest.predict(X_train)
    iso_pred_test = iso_forest.predict(X_test)

    iso_pred_train = (iso_pred_train == 1).astype(int)
    iso_pred_test = (iso_pred_test == 1).astype(int)

    # --- K‑Means Clustering (for comparison) ---
    print("  [b] K‑Means (k=2)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train)
    kmeans_labels_train = kmeans.predict(X_train)
    kmeans_labels_test = kmeans.predict(X_test)

    centers = kmeans.cluster_centers_
    norms = np.linalg.norm(centers, axis=1)
    anomaly_cluster = np.argmax(norms)
    kmeans_pred_train = (kmeans_labels_train != anomaly_cluster).astype(int)
    kmeans_pred_test = (kmeans_labels_test != anomaly_cluster).astype(int)

    # --- Evaluation ---
    def evaluate(preds, truths, name):
        acc = accuracy_score(truths, preds)
        prec = precision_score(truths, preds, pos_label=1)
        rec = recall_score(truths, preds, pos_label=1)
        f1 = f1_score(truths, preds, pos_label=1)
        print(f"\n{name} on Test Set:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1‑Score:  {f1:.4f}")
        return acc, prec, rec, f1

    print("\n" + "="*60)
    print("MODEL EVALUATION (TEST SET)")
    print("="*60)

    iso_metrics = evaluate(iso_pred_test, y_test, "Isolation Forest")
    kmeans_metrics = evaluate(kmeans_pred_test, y_test, "K‑Means")

    cm_iso = confusion_matrix(y_test, iso_pred_test)
    cm_kmeans = confusion_matrix(y_test, kmeans_pred_test)

    return (
        iso_pred_train, iso_pred_test, iso_scores_train, iso_scores_test,
        kmeans_pred_train, kmeans_pred_test, kmeans_labels_train, kmeans_labels_test,
        iso_metrics, kmeans_metrics,
        cm_iso, cm_kmeans, iso_forest, kmeans  # Return trained models!
    )


# --- HYBRID ANOMALY SCORE (INNOVATION) ---
def compute_hybrid_score(iso_scores, kmeans_proximity, alpha=0.6):
    """
    Hybrid anomaly score combining:
      - Isolation Forest score (already signed)
      - KMeans distance‑to‑centroid (convert to anomaly‑like score)

    alpha controls weight of Isolation Forest (0.0–1.0).
    """
    # Normalize both scores to roughly [0,1] scale
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
    # Convert to 0–1 (0 = normal, 1 = anomaly)
    iso_anomaly = 1.0 - iso_norm

    kmeans_norm = (kmeans_proximity - kmeans_proximity.min()) / (
        kmeans_proximity.max() - kmeans_proximity.min() + 1e-8
    )
    kmeans_anomaly = kmeans_norm  # higher distance → more anomaly‑like

    # Fuse
    hybrid = alpha * iso_anomaly + (1 - alpha) * kmeans_anomaly
    return hybrid, iso_anomaly, kmeans_anomaly


def compute_distance_to_centroids(X, kmeans_model):
    """Compute distance of each point to its nearest centroid."""
    centroids = kmeans_model.cluster_centers_
    distances = np.min(
        np.linalg.norm(X[:, np.newaxis] - centroids, axis=2),
        axis=1
    )
    return distances


# --- VISUALIZATION ---
def plot_anomalies(iso_scores_test, y_test, kmeans_proximity_test, kmeans_pred_test):
    """Plot anomaly vs normal traffic and model comparison."""
    print("[6/7] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Isolation Forest scores
    axes[0, 0].scatter(range(len(iso_scores_test)), iso_scores_test, c=y_test, cmap="RdYlGn", alpha=0.6)
    axes[0, 0].set_title("Isolation Forest Decision Scores (Green: Normal; Red: Anomaly)")
    axes[0, 0].set_ylabel("Anomaly Score")

    # 2. KMeans distance‑to‑centroid (higher = more anomaly)
    axes[0, 1].scatter(range(len(kmeans_proximity_test)), kmeans_proximity_test, c=y_test, cmap="RdYlGn", alpha=0.6)
    axes[0, 1].set_title("KMeans Distance‑to‑Centroid (Green: Normal; Red: Anomaly)")
    axes[0, 1].set_ylabel("Distance to Centroid")

    # 3. Hybrid vs Isolation Forest
    hybrid_score, _, _ = compute_hybrid_score(iso_scores_test, kmeans_proximity_test, alpha=0.6)
    axes[1, 0].scatter(range(len(hybrid_score)), hybrid_score, c=y_test, cmap="RdYlGn", alpha=0.6)
    axes[1, 0].set_title("Hybrid Anomaly Score (0–1; Green: Normal; Red: Anomaly)")
    axes[1, 0].set_ylabel("Hybrid Score")

    # 4. Confusion Matrix (Isolation Forest)
    cm_iso = confusion_matrix(y_test, (hybrid_score > 0.5).astype(int))
    sns.heatmap(cm_iso, annot=True, fmt="d", ax=axes[1, 1], cmap="Blues")
    axes[1, 1].set_title("Hybrid Confusion Matrix (Threshold = 0.5)")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig("output/anomaly_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Visualization saved to: output/anomaly_visualization.png")


# --- REAL‑TIME SIMULATION (SIMULATED STREAM) ---
def simulate_real_time_detection(X_test, iso_forest, kmeans_model, scaler, threshold=0.5):
    """Simulate incoming traffic and classify as NORMAL/ANOMALY."""
    print("[7/7] Real‑time‑like simulation started...")

    # Select a small subset of test data to simulate streaming
    n_samples = min(100, len(X_test))
    X_stream = X_test[:n_samples]

    # Inverse transform back to original‑scale values for display
    X_original = scaler.inverse_transform(X_stream)

    iso_scores_stream = iso_forest.decision_function(X_stream)
    kmeans_proximity_stream = compute_distance_to_centroids(X_stream, kmeans_model)
    hybrid_stream, _, _ = compute_hybrid_score(iso_scores_stream, kmeans_proximity_stream, alpha=0.6)

    print("\n--- Real‑Time Simulation (Sample Packets) ---")
    for i in range(n_samples):
        score = hybrid_stream[i]
        label = "NORMAL" if score > threshold else "ANOMALY"
        # Show a few key features (duration, src_bytes, protocol)
        dur = X_original[i, 0]
        src_b = X_original[i, 4]
        dst_b = X_original[i, 5]
        proto = int(X_stream[i, -3])  # protocol_type_enc (you can decode if needed)

        print(
            f"Packet {i+1}: dur={dur:.1f}, src_bytes={int(src_b)}, dst_bytes={int(dst_b)}, "
            f"proto_enc={proto} → {label} (Hybrid={score:.3f})"
        )

        if i % 10 == 0:
            print()  # newline every 10 packets


# --- MAIN PIPELINE ---
def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)

    # 1. Load and preprocess
    X, y, _ = load_and_preprocess()

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 4. Train models and evaluate
    (
        iso_pred_train, iso_pred_test, iso_scores_train, iso_scores_test,
        kmeans_pred_train, kmeans_pred_test, kmeans_labels_train, kmeans_labels_test,
        iso_metrics, kmeans_metrics,
        cm_iso, cm_kmeans, iso_forest, kmeans  # Receive trained models
    ) = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # 5. Hybrid anomaly score
    kmeans_proximity_test = compute_distance_to_centroids(X_test_scaled, kmeans)
    hybrid_score_test, iso_comp, km_comp = compute_hybrid_score(iso_scores_test, kmeans_proximity_test, alpha=0.6)

    # 6. Visualize
    plot_anomalies(iso_scores_test, y_test, kmeans_proximity_test, kmeans_pred_test)

    # 7. Real‑time simulation
    print("\n[+] Running real‑time‑like simulation with 100 samples...\n")
    simulate_real_time_detection(X_test_scaled, iso_forest, kmeans, scaler, threshold=0.5)

    print("\n[+] Project pipeline completed. Check output/anomaly_visualization.png")


if __name__ == "__main__":
    main()