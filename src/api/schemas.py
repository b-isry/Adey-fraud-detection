"""
Pydantic schemas for the Fraud Detection API.

Defines request and response models with validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np


class PredictionRequest(BaseModel):
    """Schema for prediction requests."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    customer_id: str = Field(..., description="Customer identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Transaction timestamp")
    
    # Additional features
    location: Optional[Dict[str, float]] = Field(None, description="Geographic location (lat, lon)")
    device_info: Optional[Dict[str, Any]] = Field(None, description="Device information")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    
    # Custom features
    features: Optional[Dict[str, Union[float, int, str]]] = Field(
        default_factory=dict,
        description="Additional custom features"
    )
    
    @validator('amount')
    def validate_amount(cls, v):
        """Validate transaction amount."""
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:  # $1M limit
            raise ValueError('Amount exceeds maximum limit')
        return v
    
    @validator('location')
    def validate_location(cls, v):
        """Validate location coordinates."""
        if v is not None:
            if 'lat' not in v or 'lon' not in v:
                raise ValueError('Location must contain lat and lon')
            if not (-90 <= v['lat'] <= 90):
                raise ValueError('Latitude must be between -90 and 90')
            if not (-180 <= v['lon'] <= 180):
                raise ValueError('Longitude must be between -180 and 180')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_123456789",
                "amount": 150.50,
                "merchant_id": "merchant_001",
                "customer_id": "customer_123",
                "timestamp": "2024-01-15T10:30:00Z",
                "location": {"lat": 40.7128, "lon": -74.0060},
                "device_info": {"type": "mobile", "os": "iOS"},
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0...",
                "features": {"hour_of_day": 10, "day_of_week": 1}
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    
    transaction_id: str = Field(..., description="Transaction identifier")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability score")
    fraud_prediction: bool = Field(..., description="Fraud prediction (True/False)")
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    # Additional information
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    recommendations: Optional[List[str]] = Field(None, description="Recommended actions")
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        """Validate risk level."""
        valid_levels = ['LOW', 'MEDIUM', 'HIGH']
        if v not in valid_levels:
            raise ValueError(f'Risk level must be one of: {valid_levels}')
        return v
    
    @validator('fraud_probability')
    def validate_probability(cls, v):
        """Validate probability score."""
        if not 0 <= v <= 1:
            raise ValueError('Probability must be between 0 and 1')
        return round(v, 4)
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_123456789",
                "fraud_probability": 0.0234,
                "fraud_prediction": False,
                "risk_level": "LOW",
                "confidence_score": 0.95,
                "model_version": "v1.0.0",
                "prediction_timestamp": "2024-01-15T10:30:01Z",
                "feature_importance": {"amount": 0.3, "location": 0.2},
                "explanation": "Transaction appears normal based on amount and location patterns",
                "recommendations": ["Monitor for similar patterns", "Standard processing"]
            }
        }


class ModelInfo(BaseModel):
    """Schema for model information."""
    
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    algorithm: str = Field(..., description="Algorithm used")
    training_date: datetime = Field(..., description="Training date")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    feature_count: int = Field(..., description="Number of features")
    is_active: bool = Field(..., description="Whether model is active")
    
    # Additional metadata
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")
    training_data_info: Optional[Dict[str, Any]] = Field(None, description="Training data information")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "fraud_detection_v1",
                "model_version": "v1.0.0",
                "algorithm": "RandomForest",
                "training_date": "2024-01-01T00:00:00Z",
                "performance_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90
                },
                "feature_count": 25,
                "is_active": True,
                "hyperparameters": {"n_estimators": 100, "max_depth": 10},
                "feature_names": ["amount", "location", "time_features"],
                "training_data_info": {"sample_count": 100000, "fraud_rate": 0.02}
            }
        }


class HealthCheck(BaseModel):
    """Schema for health check responses."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    
    # Component health
    model_status: str = Field(..., description="Model service status")
    database_status: Optional[str] = Field(None, description="Database status")
    cache_status: Optional[str] = Field(None, description="Cache status")
    
    # Performance metrics
    response_time_ms: Optional[float] = Field(None, description="Average response time in milliseconds")
    requests_per_minute: Optional[float] = Field(None, description="Requests per minute")
    error_rate: Optional[float] = Field(None, description="Error rate percentage")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "v1.0.0",
                "uptime": 86400.5,
                "model_status": "healthy",
                "database_status": "healthy",
                "cache_status": "healthy",
                "response_time_ms": 45.2,
                "requests_per_minute": 120.5,
                "error_rate": 0.01
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    
    transactions: List[PredictionRequest] = Field(..., description="List of transactions")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        """Validate transaction list."""
        if not v:
            raise ValueError('At least one transaction is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 transactions per batch')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123",
                "transactions": [
                    {
                        "transaction_id": "txn_1",
                        "amount": 100.0,
                        "merchant_id": "merchant_001",
                        "customer_id": "customer_123"
                    },
                    {
                        "transaction_id": "txn_2",
                        "amount": 250.0,
                        "merchant_id": "merchant_002",
                        "customer_id": "customer_456"
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""
    
    batch_id: str = Field(..., description="Batch identifier")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_summary: Dict[str, Any] = Field(..., description="Batch summary statistics")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123",
                "predictions": [
                    {
                        "transaction_id": "txn_1",
                        "fraud_probability": 0.01,
                        "fraud_prediction": False,
                        "risk_level": "LOW"
                    }
                ],
                "batch_summary": {
                    "total_transactions": 2,
                    "fraud_count": 0,
                    "average_risk_score": 0.015
                },
                "processing_time_ms": 150.5
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "error_message": "Invalid transaction amount",
                "timestamp": "2024-01-15T10:30:00Z",
                "details": {"field": "amount", "value": -100},
                "request_id": "req_123456"
            }
        } 