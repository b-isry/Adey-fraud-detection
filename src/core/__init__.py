"""
Core module for the Fraud Detection System.

Contains configuration management, logging setup, and custom exceptions.
"""

from .config import Config
from .logging import setup_logging
from .exceptions import (
    FraudDetectionError,
    DataValidationError,
    ModelTrainingError,
    PredictionError
)

__all__ = [
    "Config",
    "setup_logging", 
    "FraudDetectionError",
    "DataValidationError",
    "ModelTrainingError",
    "PredictionError"
] 