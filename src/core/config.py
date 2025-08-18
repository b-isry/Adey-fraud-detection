"""
Configuration management for the Fraud Detection System.

Handles environment-specific settings, model parameters, and system configuration.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    algorithm: str = "random_forest"
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    threshold: float = 0.5


@dataclass
class DataConfig:
    """Data processing configuration."""
    input_path: str = "data/raw/"
    processed_path: str = "data/processed/"
    features_path: str = "data/features/"
    target_column: str = "fraud"
    categorical_columns: list = field(default_factory=list)
    numerical_columns: list = field(default_factory=list)
    drop_columns: list = field(default_factory=list)


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    timeout: int = 30
    rate_limit: int = 100


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    theme: str = "light"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/fraud_detection.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class for the Fraud Detection System."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or "config/config.yaml"
        self._load_config()
        
        # Initialize sub-configurations
        self.model = ModelConfig(**self._config.get("model", {}))
        self.data = DataConfig(**self._config.get("data", {}))
        self.api = APIConfig(**self._config.get("api", {}))
        self.dashboard = DashboardConfig(**self._config.get("dashboard", {}))
        self.logging = LoggingConfig(**self._config.get("logging", {}))
        
        # Environment-specific overrides
        self._apply_environment_overrides()
    
    def _load_config(self):
        """Load configuration from file or use defaults."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                self._config = {}
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            self._config = {}
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Model overrides
        if os.getenv("MODEL_ALGORITHM"):
            self.model.algorithm = os.getenv("MODEL_ALGORITHM")
        if os.getenv("MODEL_THRESHOLD"):
            self.model.threshold = float(os.getenv("MODEL_THRESHOLD"))
        
        # API overrides
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        
        # Dashboard overrides
        if os.getenv("DASHBOARD_HOST"):
            self.dashboard.host = os.getenv("DASHBOARD_HOST")
        if os.getenv("DASHBOARD_PORT"):
            self.dashboard.port = int(os.getenv("DASHBOARD_PORT"))
        
        # Logging overrides
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "api": self.api.__dict__,
            "dashboard": self.dashboard.__dict__,
            "logging": self.logging.__dict__
        }
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate model config
            assert 0 < self.model.test_size < 1, "test_size must be between 0 and 1"
            assert 0 < self.model.validation_size < 1, "validation_size must be between 0 and 1"
            assert self.model.threshold >= 0, "threshold must be non-negative"
            
            # Validate API config
            assert 1024 <= self.api.port <= 65535, "API port must be between 1024 and 65535"
            assert self.api.timeout > 0, "API timeout must be positive"
            
            # Validate dashboard config
            assert 1024 <= self.dashboard.port <= 65535, "Dashboard port must be between 1024 and 65535"
            
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
config = Config() 