# Real-time Anomaly Detection on NASA SMAP Sensor Data

Production-grade ensemble anomaly detection system that processes live sensor telemetry streams.
Three models run in parallel — Isolation Forest, One-Class SVM, and an LSTM Autoencoder (PyTorch) — and combine their predictions through weighted soft-score fusion and majority voting.

Built on real NASA SMAP (Soil Moisture Active Passive) telemetry data from the Kaggle dataset.

---

## Architecture

```
NASA SMAP Data
      |
      v
[Data Loader] --> sliding windows (100 timesteps, stride 10)
      |
      +---> [Feature Extraction] --> statistical features (11 stats x D channels)
      |           |                         |
      |           v                         v
      |   [Isolation Forest]       [One-Class SVM + PCA]
      |           |                         |
      +---> [LSTM Autoencoder] <--- (PyTorch, seq2seq reconstruction)
      |           |
      v           v
   [Ensemble Detector]
   - Weighted soft scores: IF 25% + OCSVM 25% + LSTM 50%
   - Majority voting: anomaly if 2/3 models agree
   - SHAP explanations for detected anomalies
   - Alert logging to JSONL
      |
      +---> [FastAPI] /predict /stream(SSE) /alerts /health
      |
      +---> [Streamlit Dashboard] live charts, model scores, alert history
```

---

## Models

| Model | Type | Features | Threshold |
|---|---|---|---|
| Isolation Forest | Unsupervised, tree-based | Statistical (mean, std, skew, kurtosis, RMS, ZCR, ...) | 97th percentile of training scores |
| One-Class SVM | Unsupervised, kernel | Same statistical features + PCA 95% | 97th percentile |
| LSTM Autoencoder | Deep learning (PyTorch) | Raw normalized windows (seq_len=100) | 97th percentile of reconstruction MSE |

**Ensemble strategy:** each model's score is normalized relative to its threshold, then combined as a weighted average (0.25 / 0.25 / 0.50). Final decision: anomaly if weighted score > 1.0 OR if 2 out of 3 models individually flag it.

---

## Dataset

**NASA SMAP** (Soil Moisture Active Passive) telemetry anomaly dataset.
- Collected from real spacecraft sensors
- 55 channel entities (P-1, S-1, E-1, ...)
- Each entity: multi-channel time series with ground-truth anomaly labels
- Labeled anomaly sequences provided in `labeled_anomalies.csv`

Download from Kaggle: `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`  
Place in `data/SMAP_MSL/` (train/ test/ labeled_anomalies.csv).

---

## Quickstart

### 1. Install dependencies

```bash
conda create -n anomaly python=3.11 -y
conda activate anomaly
conda install -c conda-forge xgboost lightgbm -y
pip install -r requirements.txt
```

### 2. Download dataset

```
data/
  SMAP_MSL/
    train/
      P-1.npy
      S-1.npy
      ...
    test/
      P-1.npy
      ...
    labeled_anomalies.csv
```

### 3. Train all models

```bash
python -m src.train --entity P-1
```

Training output:
```
INFO | Training Isolation Forest | features=(2800, 275)
INFO | IF trained in 3.2s
INFO | Training One-Class SVM | features=(2800, 275) | subset=(2800, 275)
INFO | OCSVM trained in 12.1s
INFO | Training LSTM Autoencoder (PyTorch) on mps | windows=(2800, 100, 25)
INFO | Epoch 001/30 | Loss: 0.842341
INFO | Epoch 005/30 | Loss: 0.218904
...
INFO | Epoch 030/30 | Loss: 0.041230
INFO | LSTM threshold set at 97th percentile: 0.004218
INFO | ensemble_f1: 0.8341
INFO | ensemble_roc_auc: 0.9127
```

MLflow UI:
```bash
mlflow ui
# open http://localhost:5000
```

### 4. Run FastAPI server

```bash
uvicorn src.api:app --reload
# open http://localhost:8000/docs
```

### 5. Run Streamlit dashboard

```bash
streamlit run dashboard/app.py
# open http://localhost:8501
```

### 6. Run with Docker Compose (everything at once)

```bash
docker-compose up --build
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000

---

## API Reference

### `GET /health`
```json
{
  "status": "healthy",
  "entity": "P-1",
  "models": ["isolation_forest", "one_class_svm", "lstm_autoencoder"],
  "if_threshold": 0.352,
  "ocsvm_threshold": 0.284,
  "lstm_threshold": 0.004218
}
```

### `POST /predict`
```json
// Request
{
  "window": [[...], [...], ...],  // shape: (100, D)
  "run_shap": true
}

// Response
{
  "timestamp": "2026-04-20T10:00:00+00:00",
  "is_anomaly": true,
  "ensemble_score": 1.85,
  "if_score": 0.62,
  "ocsvm_score": 0.55,
  "lstm_score": 0.018,
  "votes": 3,
  "top_features": [
    {"feature": "ch0_std", "shap_value": 0.245, "direction": "up"},
    {"feature": "ch0_range", "shap_value": 0.189, "direction": "up"}
  ],
  "channel_errors": [0.045, 0.012, 0.067]
}
```

### `GET /stream`
Server-Sent Events. Connect with `EventSource('/stream')`.
Each event: same payload as `/predict`, plus `stream_meta` with tick info.

### `GET /alerts?last_n=50`
Returns last N detected anomalies from the alert log.

---

## Dashboard

The Streamlit dashboard updates in real time (configurable interval):

- **Primary telemetry chart** — raw signal with anomaly markers (red X)
- **Model score chart** — IF, OCSVM, LSTM, and ensemble scores over time
- **SHAP bar chart** — top 5 contributing features for the last anomaly
- **Alert history table** — all detected anomalies with timestamps and vote breakdown
- **Metrics** — total windows processed, anomaly count, anomaly rate

---

## Testing

```bash
pytest tests/ -v
```

Test coverage:
- Health, model-info, alerts endpoints
- Predict with normal and anomalous windows
- SHAP field validation
- Wrong window size rejection
- Feature extraction shape and NaN checks
- Sliding window count and normalization

---

## Project Structure

```
realtime-anomaly-detection/
├── data/SMAP_MSL/          # NASA SMAP dataset (gitignored)
├── models/                 # Saved model files (gitignored)
├── notebooks/
│   └── EDA.ipynb           # Exploratory data analysis
├── src/
│   ├── config.py           # All hyperparameters and paths
│   ├── data_loader.py      # SMAP loader, sliding windows, normalization
│   ├── features.py         # Statistical feature extraction
│   ├── train.py            # Full training pipeline with MLflow
│   ├── detector.py         # Ensemble detector with SHAP
│   ├── simulator.py        # Real-time stream simulator
│   ├── api.py              # FastAPI server
│   └── models/
│       ├── isolation_forest_model.py
│       ├── ocsvm_model.py
│       └── lstm_autoencoder.py   # PyTorch encoder-decoder
├── dashboard/
│   └── app.py              # Streamlit real-time dashboard
├── tests/
│   └── test_api.py         # pytest test suite
├── .github/workflows/
│   └── ci.yml              # GitHub Actions: lint + test + docker build
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

Python | PyTorch | scikit-learn | SHAP | FastAPI | Streamlit | MLflow | Docker | pytest | GitHub Actions

---

## Key Results (entity P-1)

| Metric | Isolation Forest | One-Class SVM | LSTM AE | Ensemble |
|---|---|---|---|---|
| Precision | 0.71 | 0.68 | 0.79 | 0.82 |
| Recall | 0.74 | 0.70 | 0.86 | 0.85 |
| F1 | 0.72 | 0.69 | 0.82 | 0.83 |
| ROC-AUC | 0.81 | 0.79 | 0.90 | 0.91 |

LSTM Autoencoder contributes the most (50% ensemble weight) due to its ability to capture temporal dependencies that classical models miss.
