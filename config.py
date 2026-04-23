# =============================================================
# config.py – Central Configuration for the DPI Anomaly Detector
# =============================================================
# All tuneable parameters live here so you never need to hunt
# through the source code to change a setting.
# =============================================================

import os

# ------------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models_saved")

# NSL-KDD dataset file (place KDDTrain+.txt inside the data/ folder)
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_DATA_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")  # optional

# ------------------------------------------------------------------
# 2. NSL-KDD COLUMN DEFINITIONS
# ------------------------------------------------------------------
# The NSL-KDD dataset ships WITHOUT a header row. We define every
# column name here so pandas can load them correctly.
NSL_KDD_COLUMNS = [
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

# Categorical columns that need encoding
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Numeric behavioural features we extract for the models
BEHAVIORAL_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "total_bytes", "byte_ratio", "log_src_bytes", "log_dst_bytes",
    "is_heavy_traffic", "error_rate_sum", "srv_ratio", "host_srv_ratio"
]

# ------------------------------------------------------------------
# 3. MODEL HYPER-PARAMETERS
# ------------------------------------------------------------------
# Isolation Forest
ISO_CONTAMINATION = 0.30       # expected proportion of anomalies
ISO_N_ESTIMATORS = 150         # number of trees
ISO_RANDOM_STATE = 42

# K-Means
KMEANS_N_CLUSTERS = 2          # normal vs anomaly
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 42

# Hybrid anomaly scorer
HYBRID_ALPHA = 0.6             # 0–1, weight of Isolation Forest score
HYBRID_THRESHOLD = 0.50        # packets with score > threshold → ANOMALY

# ------------------------------------------------------------------
# 4. TRAINING
# ------------------------------------------------------------------
TEST_SIZE = 0.30
SPLIT_RANDOM_STATE = 42

# ------------------------------------------------------------------
# 5. REAL-TIME SIMULATION
# ------------------------------------------------------------------
SIMULATION_N_PACKETS = 50      # how many samples to stream
SIMULATION_DELAY_SEC = 0.0     # seconds between packets (0 = instant)

# ------------------------------------------------------------------
# 6. FLASK DASHBOARD
# ------------------------------------------------------------------
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True
