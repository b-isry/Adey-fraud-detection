"""
API module for the Fraud Detection System.

Contains FastAPI-based REST API for real-time fraud detection.
"""

from .fastapi_app import create_app
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    HealthCheck
)

__all__ = [
    "create_app",
    "PredictionRequest",
    "PredictionResponse", 
    "ModelInfo",
    "HealthCheck"
] 