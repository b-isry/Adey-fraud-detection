"""
Fraud Detection System - Financial Risk Management Capstone

A comprehensive fraud detection and risk management system designed for
financial institutions with enterprise-grade reliability and transparency.
"""

__version__ = "1.0.0"
__author__ = "Bisrat Teshome"
__email__ = "bisratt1995@gmail.com"

from .core.config import Config
from .core.logging import setup_logging

# Initialize configuration and logging
config = Config()
logger = setup_logging()

__all__ = [
    "Config",
    "setup_logging",
    "config",
    "logger"
] 