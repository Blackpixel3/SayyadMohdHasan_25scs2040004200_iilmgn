# AI-Based Behavioral Deep Packet Inspection (DPI) for Anomaly Detection

A modular Python system that uses **Isolation Forest**, **K-Means Clustering**, and a novel **Hybrid Anomaly Scorer** to detect malicious network traffic using the NSL-KDD dataset.

---

## 🏗️ Project Structure

```
project/
├── main.py                          # Master pipeline (run this first)
├── app.py                           # Flask web dashboard
├── config.py                        # Central configuration
├── requirements.txt                 # Python dependencies
│
├── src/                             # Core modules
│   ├── data_loader.py               # Dataset loading & cleaning
│   ├── preprocessor.py              # Encoding, splitting, scaling
│   ├── feature_engineering.py       # Derived feature creation
│   ├── evaluation.py                # Metrics & comparison report
│   ├── visualization.py             # Publication-quality charts
│   ├── realtime_simulator.py        # Simulated packet stream
│   └── models/
│       ├── isolation_forest_model.py
│       ├── kmeans_model.py
│       └── hybrid_detector.py       # ← Innovation component
│
├── data/                            # Place dataset here
│   └── KDDTrain+.txt
│
├── output/                          # Generated plots (auto-created)
└── models_saved/                    # Saved trained models (auto-created)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the NSL-KDD Dataset

Download from: https://www.unb.ca/cic/datasets/nsl.html

Place `KDDTrain+.txt` inside the `data/` folder.

### 3. Run the Pipeline

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Engineer behavioural features
- Train Isolation Forest and K-Means models
- Compute hybrid anomaly scores
- Evaluate all models and print a comparison report
- Generate 5 visualisation charts in `output/`
- Run a real-time packet inspection simulation
- Save trained models to `models_saved/`

### 4. Launch the Web Dashboard (Optional)

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

---

## 🧠 Innovation: Hybrid Anomaly Detector

The key innovation is the **Hybrid Anomaly Scorer** (`src/models/hybrid_detector.py`):

1. **Score Fusion**: Combines normalised Isolation Forest decision scores with K-Means centroid distances using a weighted formula:
   ```
   hybrid = α × IF_anomaly + (1 − α) × KM_anomaly
   ```

2. **Adaptive Threshold**: Rather than a fixed cut-off, the threshold dynamically adjusts to `mean + 1.5 × std` of the score distribution, making the detector robust across different traffic patterns.

---

## 📊 Models

| Model | Type | Approach |
|-------|------|----------|
| Isolation Forest | Unsupervised | Isolates anomalies via random partitioning |
| K-Means (k=2) | Clustering | Distance from nearest cluster centroid |
| Hybrid Fusion | Ensemble | Weighted combination + adaptive threshold |

---

## 📈 Evaluation Metrics

- **Accuracy** — overall correctness
- **Precision** — of predicted anomalies, how many are real?
- **Recall** — of real anomalies, how many did we catch?
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — detailed true/false positive/negative counts

---

## 🔧 Configuration

All tuneable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ISO_CONTAMINATION` | 0.30 | Expected proportion of anomalies |
| `ISO_N_ESTIMATORS` | 150 | Number of Isolation Forest trees |
| `KMEANS_N_CLUSTERS` | 2 | Number of clusters |
| `HYBRID_ALPHA` | 0.6 | Weight of IF component in fusion |
| `HYBRID_THRESHOLD` | 0.50 | Base detection threshold |
| `TEST_SIZE` | 0.30 | Train/test split ratio |

---

## 📝 License

This project is for educational and research purposes.
