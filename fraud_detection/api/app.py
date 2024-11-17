# fraud_detection/api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List

app = FastAPI()

class TransactionInput(BaseModel):
    amount: float
    merchant_category: str
    merchant_type: str
    channel: str
    device: str
    distance_from_home: int
    high_risk_merchant: bool
    transaction_hour: int
    weekend_transaction: bool
    velocity_num_transactions: int
    velocity_total_amount: float
    velocity_unique_merchants: int

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_factors: List[str]

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Generate predictions (assuming model is loaded)
        fraud_prob = 0.5  # Replace with actual model prediction
        
        # Identify risk factors
        risk_factors = []
        if transaction.high_risk_merchant:
            risk_factors.append("High risk merchant")
        if transaction.amount > 1000:
            risk_factors.append("Large transaction amount")
        if transaction.distance_from_home > 0:
            risk_factors.append("Transaction away from home")
        
        return PredictionResponse(
            fraud_probability=fraud_prob,
            is_fraud=fraud_prob > 0.5,
            risk_factors=risk_factors
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
