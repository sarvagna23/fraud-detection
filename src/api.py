from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
import uuid
from database import init_db, log_transaction

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_model.pkl')

app = FastAPI(title="Fraud Detection API", version="1.0.0")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

init_db()

class Transaction(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features")

    features = np.array(transaction.features).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]
    is_fraud = bool(prob >= 0.5)

    if prob >= 0.8:
        risk = "HIGH"
    elif prob >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    transaction_id = str(uuid.uuid4())
    log_transaction(transaction_id, float(prob), is_fraud, risk)

    return PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=is_fraud,
        fraud_probability=round(float(prob), 4),
        risk_level=risk
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)