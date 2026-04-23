# =============================================================
# src/realtime_simulator.py - Real-Time Traffic Simulation
# =============================================================
# Simulates a stream of incoming network packets and classifies
# each one in real time using the trained Hybrid Detector.
# =============================================================

import sys
import io
import time
import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import SIMULATION_N_PACKETS, SIMULATION_DELAY_SEC


class RealTimeSimulator:
    """
    Takes a sample of test data and 'streams' it packet-by-packet,
    printing the classification result for each packet just like
    a real-time IDS (Intrusion Detection System) would.
    """

    def __init__(
        self,
        iso_model,
        kmeans_model,
        hybrid_detector,
        scaler,
        feature_names: list,
    ):
        self.iso_model = iso_model
        self.kmeans_model = kmeans_model
        self.hybrid_detector = hybrid_detector
        self.scaler = scaler
        self.feature_names = feature_names

    # ----------------------------------------------------------
    def run(
        self,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray = None,
        n_packets: int = SIMULATION_N_PACKETS,
        delay: float = SIMULATION_DELAY_SEC,
    ) -> list:
        """
        Simulate real-time packet inspection.

        Parameters
        ----------
        X_test_scaled : np.ndarray
            Scaled test feature matrix.
        y_test : np.ndarray, optional
            True labels (for printing ground truth alongside prediction).
        n_packets : int
            Number of packets to simulate.
        delay : float
            Seconds to wait between packets (0 = no delay).

        Returns
        -------
        list[dict]
            List of packet result dicts (useful for the Flask dashboard).
        """
        n_packets = min(n_packets, len(X_test_scaled))
        X_stream = X_test_scaled[:n_packets]

        # Get scores from both models
        iso_scores = self.iso_model.decision_scores(X_stream)
        km_distances = self.kmeans_model.centroid_distances(X_stream)

        # Hybrid prediction
        predictions, hybrid_scores, threshold = self.hybrid_detector.predict(
            iso_scores, km_distances, use_adaptive=True
        )

        # Inverse-transform for display (original scale)
        X_original = self.scaler.inverse_transform(X_stream)

        print("\n" + "=" * 72)
        print("REAL-TIME PACKET INSPECTION SIMULATION")
        print(f"  Packets to inspect : {n_packets}")
        print(f"  Adaptive threshold : {threshold:.4f}")
        print("=" * 72)

        results_list = []
        normal_count = 0
        anomaly_count = 0

        for i in range(n_packets):
            label = "NORMAL" if predictions[i] == 1 else "ANOMALY"
            score = hybrid_scores[i]

            # Extract a few key features for display
            duration = X_original[i, 0]
            src_bytes = int(X_original[i, 1])
            dst_bytes = int(X_original[i, 2])

            # Ground truth (if available)
            gt = ""
            if y_test is not None:
                gt_label = "NORMAL" if y_test[i] == 1 else "ANOMALY"
                gt = f"  [GT: {gt_label}]"
                correct = "Y" if (predictions[i] == y_test[i]) else "N"
                gt += f"  {correct}"

            # Count
            if predictions[i] == 1:
                normal_count += 1
            else:
                anomaly_count += 1

            # Build result dict
            pkt_result = {
                "id": i + 1,
                "duration": round(float(duration), 2),
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "hybrid_score": round(float(score), 4),
                "label": label,
            }
            results_list.append(pkt_result)

            # Console output
            status_icon = "[OK]" if label == "NORMAL" else "[!!]"
            print(
                f"  {status_icon} Pkt #{i+1:>4d}  | "
                f"dur={duration:>8.1f}  src={src_bytes:>7d}B  dst={dst_bytes:>7d}B  | "
                f"score={score:.4f}  -> {label:<8s}{gt}"
            )

            if delay > 0:
                time.sleep(delay)

        # Summary
        print("\n" + "-" * 72)
        print(f"  SIMULATION SUMMARY")
        print(f"    Total packets   : {n_packets}")
        print(f"    Normal          : {normal_count} ({100*normal_count/n_packets:.1f}%)")
        print(f"    Anomalous       : {anomaly_count} ({100*anomaly_count/n_packets:.1f}%)")
        print(f"    Threshold used  : {threshold:.4f}")
        print("-" * 72 + "\n")

        return results_list
