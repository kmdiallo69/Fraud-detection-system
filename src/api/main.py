#!/usr/bin/env python3
"""
Fraud Detection API
==================

FastAPI application for serving fraud detection predictions.
Provides REST API endpoints for real-time and batch predictions.

Author: ML Engineer
Date: 2024
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import FraudPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent credit card transactions using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None


# Pydantic models for request/response
class TransactionData(BaseModel):
    """Model for single transaction data."""

    V1: float = Field(..., description="Feature V1")
    V2: float = Field(..., description="Feature V2")
    V3: float = Field(..., description="Feature V3")
    V4: float = Field(..., description="Feature V4")
    V5: float = Field(..., description="Feature V5")
    V6: float = Field(..., description="Feature V6")
    V7: float = Field(..., description="Feature V7")
    V8: float = Field(..., description="Feature V8")
    V9: float = Field(..., description="Feature V9")
    V10: float = Field(..., description="Feature V10")
    V11: float = Field(..., description="Feature V11")
    V12: float = Field(..., description="Feature V12")
    V13: float = Field(..., description="Feature V13")
    V14: float = Field(..., description="Feature V14")
    V15: float = Field(..., description="Feature V15")
    V16: float = Field(..., description="Feature V16")
    V17: float = Field(..., description="Feature V17")
    V18: float = Field(..., description="Feature V18")
    V19: float = Field(..., description="Feature V19")
    V20: float = Field(..., description="Feature V20")
    V21: float = Field(..., description="Feature V21")
    V22: float = Field(..., description="Feature V22")
    V23: float = Field(..., description="Feature V23")
    V24: float = Field(..., description="Feature V24")
    V25: float = Field(..., description="Feature V25")
    V26: float = Field(..., description="Feature V26")
    V27: float = Field(..., description="Feature V27")
    V28: float = Field(..., description="Feature V28")
    Amount: float = Field(..., description="Transaction amount")
    hour: int = Field(..., description="Hour of transaction")
    day: int = Field(..., description="Day of transaction")
    amount_log: float = Field(..., description="Log of amount")
    amount_sqrt: float = Field(..., description="Square root of amount")
    v_mean: float = Field(..., description="Mean of V features")
    v_std: float = Field(..., description="Standard deviation of V features")
    v_max: float = Field(..., description="Maximum of V features")
    v_min: float = Field(..., description="Minimum of V features")
    v_range: float = Field(..., description="Range of V features")
    amount_time_interaction: float = Field(..., description="Amount-time interaction")
    Amount_capped: float = Field(..., description="Capped amount")


class PredictionResponse(BaseModel):
    """Model for prediction response."""

    prediction: int = Field(..., description="Prediction (0=Normal, 1=Fraud)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., description="Confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_info: Dict = Field(..., description="Model information")


class BatchPredictionRequest(BaseModel):
    """Model for batch prediction request."""

    transactions: List[TransactionData] = Field(..., description="List of transactions")


class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response."""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    total_transactions: int = Field(..., description="Total number of transactions")
    fraud_count: int = Field(..., description="Number of fraud predictions")
    normal_count: int = Field(..., description="Number of normal predictions")
    timestamp: str = Field(..., description="Batch prediction timestamp")


class HealthResponse(BaseModel):
    """Model for health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Dict = Field(..., description="Model information")
    timestamp: str = Field(..., description="Health check timestamp")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global predictor
    try:
        logger.info("🚀 Starting Fraud Detection API...")
        predictor = FraudPredictor()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "batch_predict": "/batch_predict",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model_info = predictor.get_model_info() if predictor else {}
        return HealthResponse(
            status="healthy",
            model_loaded=predictor is not None,
            model_info=model_info,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: TransactionData):
    """
    Make a prediction for a single transaction.

    Args:
        transaction: Transaction data

    Returns:
        Prediction result
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert Pydantic model to dict
        transaction_dict = transaction.dict()

        # Make prediction
        result = predictor.predict_single(transaction_dict)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest, background_tasks: BackgroundTasks
):
    """
    Make predictions for multiple transactions.

    Args:
        request: Batch prediction request
        background_tasks: Background tasks for saving results

    Returns:
        Batch prediction results
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert Pydantic models to dicts
        transactions = [t.dict() for t in request.transactions]

        # Make predictions
        results = predictor.predict_batch(transactions)

        # Filter out errors
        valid_predictions = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]

        if errors:
            logger.warning(f"Batch prediction had {len(errors)} errors")

        # Count predictions
        fraud_count = sum(1 for p in valid_predictions if p["prediction"] == 1)
        normal_count = sum(1 for p in valid_predictions if p["prediction"] == 0)

        # Save predictions in background
        if valid_predictions:
            background_tasks.add_task(predictor.save_predictions, valid_predictions)

        return BatchPredictionResponse(
            predictions=[PredictionResponse(**p) for p in valid_predictions],
            total_transactions=len(transactions),
            fraud_count=fraud_count,
            normal_count=normal_count,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info", response_model=Dict)
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        Model information
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return predictor.get_model_info()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample_transaction", response_model=TransactionData)
async def get_sample_transaction():
    """
    Get a sample transaction for testing.

    Returns:
        Sample transaction data
    """
    try:
        from predict import create_sample_transaction

        sample = create_sample_transaction()
        return TransactionData(**sample)

    except Exception as e:
        logger.error(f"Sample transaction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the API server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
