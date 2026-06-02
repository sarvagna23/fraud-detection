# Fraud Detection ML Pipeline

Enterprise-grade machine learning pipeline for real-time credit card fraud detection using XGBoost, FastAPI, PostgreSQL, and Docker.

## Results
- **Accuracy:** 99.84%
- **ROC-AUC:** 0.9788
- **PR-AUC:** 0.8572
- **Dataset:** 284,807 transactions (492 fraud cases)
- **True Positives:** 85/98 fraud cases caught

## Architecture
creditcard.csv → Preprocessing (SMOTE) → XGBoost Model → FastAPI → PostgreSQL
↓
Docker → AWS EC2

## Tech Stack
- **ML:** Python, XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **API:** FastAPI, Uvicorn
- **Database:** SQLite (local), PostgreSQL (production)
- **Infrastructure:** Docker, AWS EC2
- **Testing:** pytest (7/7 tests passing)

## Project Structure

fraud-detection/
├── src/
│   ├── ingest.py        # Data loading and validation
│   ├── preprocess.py    # Feature scaling and SMOTE oversampling
│   ├── train.py         # XGBoost model training and evaluation
│   ├── api.py           # FastAPI REST endpoint
│   └── database.py      # SQL transaction logging
├── tests/
│   └── test_pipeline.py # Automated functional tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt


## Quick Start

```bash
# Clone and setup
git clone https://github.com/sarvagna23/fraud-detection.git
cd fraud-detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in data/

# Train model
python3 src/train.py

# Start API
python3 src/api.py
```

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict fraud
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}'  # 30 features
```

Response:
```json
{
  "transaction_id": "546ed7b4-b0b6-4c45-9565-6151384b3c97",
  "is_fraud": false,
  "fraud_probability": 0.0015,
  "risk_level": "LOW"
}
```

## Docker

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## Dataset
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — MLG ULB, Kaggle