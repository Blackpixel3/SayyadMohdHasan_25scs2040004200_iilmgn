# =============================================================
# app.py – Flask Web Dashboard
# =============================================================
# A modern, real-time web dashboard that displays:
#   • Live packet inspection feed
#   • Detection statistics (normal vs anomaly counts)
#   • Model performance metrics
#   • Anomaly score distribution chart
#
# Usage:
#   1. First run:  python main.py   (to train models)
#   2. Then run:   python app.py    (to start the dashboard)
#   3. Open:       http://127.0.0.1:5000
# =============================================================

import os
import sys
import io
import json
import time
import random
import threading
import warnings

warnings.filterwarnings("ignore")

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import joblib
from flask import Flask, render_template_string, jsonify

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    MODEL_DIR, OUTPUT_DIR,
)

app = Flask(__name__)

# ---------------------------------------------------------------
# Global state (thread-safe)
# ---------------------------------------------------------------
lock = threading.Lock()
state = {
    "packets": [],        # last N processed packets
    "total_normal": 0,
    "total_anomaly": 0,
    "running": True,
    "models_loaded": False,
}

# ---------------------------------------------------------------
# Load trained models (from main.py output)
# ---------------------------------------------------------------
iso_model = None
kmeans_model = None
hybrid_model = None
scaler = None
test_data = None



def load_models():
    """Try to load saved models; fall back to simulation mode."""
    global iso_model, kmeans_model, hybrid_model, scaler, test_data

    try:
        iso_model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
        kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
        hybrid_model = joblib.load(os.path.join(MODEL_DIR, "hybrid_detector.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        
        try:
            test_data = joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))
        except FileNotFoundError:
            test_data = None
            
        state["models_loaded"] = True
        print("[Dashboard] Trained models loaded successfully ✓")
    except FileNotFoundError:
        print("[Dashboard] WARNING: Trained models not found.")
        print("            Run 'python main.py' first to train models.")
        print("            Dashboard will use simulated predictions.\n")
        state["models_loaded"] = False


# ---------------------------------------------------------------
# Background thread: simulate incoming packets
# ---------------------------------------------------------------
def packet_simulator():
    """Generate fake packets every second and classify them."""
    pkt_id = 0
    while state["running"]:
        pkt_id += 1

        if state["models_loaded"] and scaler is not None and hybrid_model is not None and test_data is not None:
            # Replay a random packet from the actual test data
            X_test_np = test_data["X_test_unscaled"]
            y_test_np = test_data["y_test"]
            
            row_idx = random.randint(0, len(X_test_np) - 1)
            raw_features = X_test_np[row_idx]
            
            duration = float(raw_features[0])
            src_bytes = int(raw_features[1])
            dst_bytes = int(raw_features[2])
            # Assuming protocol_type_enc is one of the last few columns, pick string randomly for display
            protocol = random.choice(["TCP", "UDP", "ICMP"])
            
            scaled = scaler.transform(raw_features.reshape(1, -1))
            iso_score = iso_model.decision_scores(scaled)
            km_dist = kmeans_model.centroid_distances(scaled)
            _, hybrid_score, _ = hybrid_model.predict(
                iso_score, km_dist, use_adaptive=False
            )
            score = float(hybrid_score[0])
            label = "ANOMALY" if score >= hybrid_model.base_threshold else "NORMAL"
        else:
            # Fallback: random classification
            duration = round(random.uniform(0.0, 50.0), 2)
            src_bytes = random.randint(0, 50000)
            dst_bytes = random.randint(0, 50000)
            protocol = random.choice(["TCP", "UDP", "ICMP"])
            
            score = round(random.uniform(0, 1), 4)
            label = "ANOMALY" if random.random() < 0.25 else "NORMAL"

        pkt = {
            "id": pkt_id,
            "time": time.strftime("%H:%M:%S"),
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "protocol": protocol,
            "score": round(score, 4),
            "label": label,
        }

        with lock:
            state["packets"].append(pkt)
            if len(state["packets"]) > 100:
                state["packets"] = state["packets"][-100:]
            if label == "NORMAL":
                state["total_normal"] += 1
            else:
                state["total_anomaly"] += 1

        time.sleep(1.2)


# ---------------------------------------------------------------
# HTML template (modern dark theme, auto-refreshing)
# ---------------------------------------------------------------
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DPI Anomaly Detection – Dashboard</title>
    <meta name="description" content="AI-Based Behavioral Deep Packet Inspection Dashboard for real-time network anomaly detection">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* ── Reset & Base ─────────────────────────────── */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: #1a2233;
            --bg-card-hover: #1f2a3d;
            --border: #2a3548;
            --text-primary: #e8edf5;
            --text-secondary: #8892a4;
            --text-muted: #5a6478;
            --accent-green: #00d68f;
            --accent-green-glow: rgba(0, 214, 143, 0.15);
            --accent-red: #ff4757;
            --accent-red-glow: rgba(255, 71, 87, 0.15);
            --accent-blue: #3b82f6;
            --accent-blue-glow: rgba(59, 130, 246, 0.15);
            --accent-purple: #8b5cf6;
            --accent-yellow: #fbbf24;
            --gradient-main: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
            --shadow-lg: 0 10px 30px rgba(0,0,0,0.5);
            --radius: 12px;
            --radius-sm: 8px;
        }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* ── Animated background ──────────────────────── */
        body::before {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.05) 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 50%, rgba(139,92,246,0.05) 0%, transparent 50%),
                        radial-gradient(ellipse at 50% 100%, rgba(0,214,143,0.03) 0%, transparent 50%);
            animation: bgShift 20s ease-in-out infinite alternate;
            z-index: -1;
        }
        @keyframes bgShift {
            0% { transform: translate(0, 0); }
            100% { transform: translate(-5%, -5%); }
        }

        /* ── Layout ───────────────────────────────────── */
        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }

        /* ── Header ───────────────────────────────────── */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 28px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }
        .header-left { display: flex; align-items: center; gap: 16px; }
        .logo {
            width: 48px; height: 48px;
            background: var(--gradient-main);
            border-radius: var(--radius-sm);
            display: flex; align-items: center; justify-content: center;
            font-size: 22px; font-weight: 700; color: white;
            box-shadow: 0 0 20px rgba(102,126,234,0.3);
        }
        .header h1 {
            font-size: 22px; font-weight: 700;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header p { color: var(--text-secondary); font-size: 13px; margin-top: 2px; }
        .status-badge {
            display: flex; align-items: center; gap: 8px;
            padding: 8px 16px;
            background: var(--accent-green-glow);
            border: 1px solid rgba(0,214,143,0.3);
            border-radius: 50px;
            font-size: 13px; font-weight: 500; color: var(--accent-green);
        }
        .status-dot {
            width: 8px; height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,214,143,0.4); }
            50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0,214,143,0); }
        }

        /* ── Stats grid ───────────────────────────────── */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .stat-label {
            font-size: 12px; font-weight: 500; text-transform: uppercase;
            letter-spacing: 1px; color: var(--text-muted); margin-bottom: 8px;
        }
        .stat-value {
            font-size: 32px; font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .stat-card.normal .stat-value { color: var(--accent-green); }
        .stat-card.anomaly .stat-value { color: var(--accent-red); }
        .stat-card.total .stat-value { color: var(--accent-blue); }
        .stat-card.rate .stat-value { color: var(--accent-yellow); }
        .stat-sub { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

        /* ── Main grid ────────────────────────────────── */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 360px;
            gap: 20px;
        }
        @media (max-width: 960px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        /* ── Cards ────────────────────────────────────── */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
        }
        .card-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; justify-content: space-between;
        }
        .card-title {
            font-size: 14px; font-weight: 600;
            display: flex; align-items: center; gap: 8px;
        }
        .card-body { padding: 0; }

        /* ── Packet feed ──────────────────────────────── */
        .packet-feed {
            max-height: 520px;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: var(--border) transparent;
        }
        .packet-feed::-webkit-scrollbar { width: 6px; }
        .packet-feed::-webkit-scrollbar-track { background: transparent; }
        .packet-feed::-webkit-scrollbar-thumb {
            background: var(--border); border-radius: 3px;
        }

        .packet-row {
            display: grid;
            grid-template-columns: 50px 60px 1fr 90px 90px 80px 100px;
            align-items: center;
            padding: 10px 20px;
            border-bottom: 1px solid rgba(42,53,72,0.5);
            font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
            transition: background 0.15s;
        }
        .packet-row:hover { background: var(--bg-card-hover); }
        .packet-row.header {
            background: rgba(59,130,246,0.05);
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            position: sticky; top: 0; z-index: 1;
            border-bottom: 1px solid var(--border);
        }

        .badge {
            display: inline-flex; align-items: center;
            padding: 3px 10px; border-radius: 50px;
            font-size: 11px; font-weight: 600;
            font-family: 'Inter', sans-serif;
        }
        .badge-normal {
            background: var(--accent-green-glow);
            color: var(--accent-green);
            border: 1px solid rgba(0,214,143,0.25);
        }
        .badge-anomaly {
            background: var(--accent-red-glow);
            color: var(--accent-red);
            border: 1px solid rgba(255,71,87,0.25);
            animation: flashAnomaly 1.5s ease-in-out;
        }
        @keyframes flashAnomaly {
            0% { box-shadow: 0 0 12px rgba(255,71,87,0.5); }
            100% { box-shadow: none; }
        }

        .score-bar-wrapper {
            width: 100%; height: 6px; background: rgba(255,255,255,0.06);
            border-radius: 3px; overflow: hidden;
        }
        .score-bar {
            height: 100%; border-radius: 3px;
            transition: width 0.4s ease;
        }
        .score-bar.low { background: var(--accent-green); }
        .score-bar.med { background: var(--accent-yellow); }
        .score-bar.high { background: var(--accent-red); }

        /* ── Side panels ──────────────────────────────── */
        .side-panels { display: flex; flex-direction: column; gap: 20px; }

        .chart-placeholder {
            padding: 20px; text-align: center;
        }
        .donut-container {
            width: 200px; height: 200px; margin: 0 auto 16px;
            position: relative;
        }
        .donut-container canvas { width: 100% !important; height: 100% !important; }
        .donut-center {
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }
        .donut-center .big {
            font-size: 28px; font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }
        .donut-center .sub { font-size: 11px; color: var(--text-muted); }

        /* ── Model info ───────────────────────────────── */
        .model-info { padding: 16px 20px; }
        .model-item {
            display: flex; align-items: center; gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(42,53,72,0.4);
        }
        .model-item:last-child { border-bottom: none; }
        .model-icon {
            width: 36px; height: 36px; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 16px;
        }
        .model-icon.if-icon { background: var(--accent-blue-glow); }
        .model-icon.km-icon { background: var(--accent-red-glow); }
        .model-icon.hy-icon {
            background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15));
        }
        .model-name { font-size: 13px; font-weight: 600; }
        .model-desc { font-size: 11px; color: var(--text-muted); }

        /* ── Footer ───────────────────────────────────── */
        .footer {
            text-align: center; margin-top: 32px; padding-top: 20px;
            border-top: 1px solid var(--border);
            font-size: 12px; color: var(--text-muted);
        }

        /* ── Canvas donut (manual SVG) ────────────────── */
        .svg-donut { transform: rotate(-90deg); }
        .svg-donut circle {
            fill: none; stroke-width: 14;
            stroke-linecap: round;
        }
        .svg-donut .bg-ring { stroke: rgba(255,255,255,0.04); }
        .svg-donut .normal-ring { stroke: var(--accent-green); transition: stroke-dashoffset 0.6s ease; }
        .svg-donut .anomaly-ring { stroke: var(--accent-red); transition: stroke-dashoffset 0.6s ease; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header" id="dashboard-header">
            <div class="header-left">
                <div class="logo">DPI</div>
                <div>
                    <h1>AI Behavioral DPI – Anomaly Detection</h1>
                    <p>Hybrid Isolation Forest × K-Means × Adaptive Threshold</p>
                </div>
            </div>
            <div class="status-badge" id="status-badge">
                <span class="status-dot"></span>
                <span id="status-text">Monitoring Active</span>
            </div>
        </header>

        <!-- Stats -->
        <div class="stats-grid" id="stats-grid">
            <div class="stat-card total">
                <div class="stat-label">Total Packets</div>
                <div class="stat-value" id="stat-total">0</div>
                <div class="stat-sub">Inspected so far</div>
            </div>
            <div class="stat-card normal">
                <div class="stat-label">Normal Traffic</div>
                <div class="stat-value" id="stat-normal">0</div>
                <div class="stat-sub" id="stat-normal-pct">0%</div>
            </div>
            <div class="stat-card anomaly">
                <div class="stat-label">Anomalies Detected</div>
                <div class="stat-value" id="stat-anomaly">0</div>
                <div class="stat-sub" id="stat-anomaly-pct">0%</div>
            </div>
            <div class="stat-card rate">
                <div class="stat-label">Anomaly Rate</div>
                <div class="stat-value" id="stat-rate">0%</div>
                <div class="stat-sub">Rolling average</div>
            </div>
        </div>

        <!-- Main content -->
        <div class="main-grid">
            <!-- Packet Feed -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span>📡</span> Live Packet Feed
                    </div>
                    <span style="font-size:12px;color:var(--text-muted);" id="feed-info">Auto-refreshing every 2s</span>
                </div>
                <div class="card-body">
                    <div class="packet-feed" id="packet-feed">
                        <div class="packet-row header">
                            <span>#</span>
                            <span>Time</span>
                            <span>Details</span>
                            <span>Src Bytes</span>
                            <span>Dst Bytes</span>
                            <span>Score</span>
                            <span>Verdict</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Side panels -->
            <div class="side-panels">
                <!-- Donut chart -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><span>📊</span> Traffic Distribution</div>
                    </div>
                    <div class="card-body chart-placeholder">
                        <div class="donut-container">
                            <svg class="svg-donut" viewBox="0 0 120 120" width="200" height="200">
                                <circle class="bg-ring" cx="60" cy="60" r="50"/>
                                <circle class="normal-ring" id="normal-ring" cx="60" cy="60" r="50"
                                        stroke-dasharray="314.16" stroke-dashoffset="314.16"/>
                                <circle class="anomaly-ring" id="anomaly-ring" cx="60" cy="60" r="50"
                                        stroke-dasharray="314.16" stroke-dashoffset="314.16"/>
                            </svg>
                            <div class="donut-center">
                                <div class="big" id="donut-total">0</div>
                                <div class="sub">packets</div>
                            </div>
                        </div>
                        <div style="display:flex;justify-content:center;gap:24px;font-size:13px;">
                            <span style="color:var(--accent-green);">● Normal <span id="donut-normal">0</span></span>
                            <span style="color:var(--accent-red);">● Anomaly <span id="donut-anomaly">0</span></span>
                        </div>
                    </div>
                </div>

                <!-- Models info -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><span>🧠</span> Detection Models</div>
                    </div>
                    <div class="card-body model-info">
                        <div class="model-item">
                            <div class="model-icon if-icon">🌲</div>
                            <div>
                                <div class="model-name">Isolation Forest</div>
                                <div class="model-desc">Unsupervised anomaly detection — isolates outliers via random partitioning</div>
                            </div>
                        </div>
                        <div class="model-item">
                            <div class="model-icon km-icon">🎯</div>
                            <div>
                                <div class="model-name">K-Means Clustering</div>
                                <div class="model-desc">Groups traffic into normal/anomaly clusters by centroid distance</div>
                            </div>
                        </div>
                        <div class="model-item">
                            <div class="model-icon hy-icon">⚡</div>
                            <div>
                                <div class="model-name">Hybrid Fusion (Innovation)</div>
                                <div class="model-desc">Weighted score fusion with adaptive threshold for robust detection</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer class="footer" id="dashboard-footer">
            AI-Based Behavioral Deep Packet Inspection for Anomaly Detection — Final Year Project
        </footer>
    </div>

    <script>
        const CIRC = 2 * Math.PI * 50; // circumference of r=50 circle

        async function fetchData() {
            try {
                const res = await fetch('/api/packets');
                const data = await res.json();

                // Update stats
                const total = data.total_normal + data.total_anomaly;
                document.getElementById('stat-total').textContent = total.toLocaleString();
                document.getElementById('stat-normal').textContent = data.total_normal.toLocaleString();
                document.getElementById('stat-anomaly').textContent = data.total_anomaly.toLocaleString();

                const normalPct = total > 0 ? (100 * data.total_normal / total).toFixed(1) : 0;
                const anomalyPct = total > 0 ? (100 * data.total_anomaly / total).toFixed(1) : 0;
                document.getElementById('stat-normal-pct').textContent = normalPct + '% of traffic';
                document.getElementById('stat-anomaly-pct').textContent = anomalyPct + '% of traffic';
                document.getElementById('stat-rate').textContent = anomalyPct + '%';

                // Donut
                document.getElementById('donut-total').textContent = total;
                document.getElementById('donut-normal').textContent = data.total_normal;
                document.getElementById('donut-anomaly').textContent = data.total_anomaly;

                if (total > 0) {
                    const nFrac = data.total_normal / total;
                    const aFrac = data.total_anomaly / total;
                    const normalOffset = CIRC * (1 - nFrac);
                    const anomalyLen = CIRC * aFrac;
                    const anomalyOffset = CIRC - anomalyLen;

                    document.getElementById('normal-ring').style.strokeDashoffset = normalOffset;
                    const aRing = document.getElementById('anomaly-ring');
                    aRing.style.strokeDasharray = `${anomalyLen} ${CIRC - anomalyLen}`;
                    aRing.style.strokeDashoffset = -CIRC * nFrac;
                }

                // Packet feed
                const feed = document.getElementById('packet-feed');
                // Keep header row
                const header = feed.querySelector('.packet-row.header');
                feed.innerHTML = '';
                feed.appendChild(header);

                // Show newest first
                const packets = data.packets.slice().reverse();
                packets.forEach(pkt => {
                    const row = document.createElement('div');
                    row.className = 'packet-row';

                    const scoreClass = pkt.score < 0.3 ? 'low' : pkt.score < 0.6 ? 'med' : 'high';
                    const badgeClass = pkt.label === 'NORMAL' ? 'badge-normal' : 'badge-anomaly';

                    row.innerHTML = `
                        <span style="color:var(--text-muted)">${pkt.id}</span>
                        <span style="color:var(--text-secondary)">${pkt.time}</span>
                        <span style="color:var(--text-secondary)">${pkt.protocol} · dur=${pkt.duration}s</span>
                        <span>${pkt.src_bytes.toLocaleString()}</span>
                        <span>${pkt.dst_bytes.toLocaleString()}</span>
                        <span>
                            <div class="score-bar-wrapper">
                                <div class="score-bar ${scoreClass}" style="width:${Math.round(pkt.score*100)}%"></div>
                            </div>
                            <span style="font-size:11px;color:var(--text-muted)">${pkt.score.toFixed(3)}</span>
                        </span>
                        <span><span class="badge ${badgeClass}">${pkt.label}</span></span>
                    `;
                    feed.appendChild(row);
                });

            } catch (e) {
                console.error('Fetch error:', e);
            }
        }

        // Initial fetch + auto-refresh
        fetchData();
        setInterval(fetchData, 2000);
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/packets")
def api_packets():
    """JSON API: return current packet data and stats."""
    with lock:
        return jsonify(
            packets=state["packets"][-50:],  # last 50
            total_normal=state["total_normal"],
            total_anomaly=state["total_anomaly"],
            models_loaded=state["models_loaded"],
        )


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    load_models()

    # Start background simulator
    sim_thread = threading.Thread(target=packet_simulator, daemon=True)
    sim_thread.start()
    print(f"\n[Dashboard] Starting at http://{FLASK_HOST}:{FLASK_PORT}")
    print("[Dashboard] Press Ctrl+C to stop\n")

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, use_reloader=False)