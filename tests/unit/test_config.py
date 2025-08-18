"""
Unit tests for configuration management.
"""

import pytest
import os
import tempfile
from pathlib import Path

from src.core.config import Config, ModelConfig, DataConfig, APIConfig


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.algorithm == "random_forest"
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert config.threshold == 0.5
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            algorithm="gradient_boosting",
            random_state=123,
            threshold=0.7
        )
        
        assert config.algorithm == "gradient_boosting"
        assert config.random_state == 123
        assert config.threshold == 0.7


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        
        assert config.input_path == "data/raw/"
        assert config.target_column == "fraud"
        assert config.categorical_columns == []
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataConfig(
            input_path="custom/path/",
            target_column="is_fraud",
            categorical_columns=["category1", "category2"]
        )
        
        assert config.input_path == "custom/path/"
        assert config.target_column == "is_fraud"
        assert config.categorical_columns == ["category1", "category2"]


class TestAPIConfig:
    """Test APIConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = APIConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = APIConfig(
            host="127.0.0.1",
            port=9000,
            debug=True
        )
        
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.debug is True


class TestConfig:
    """Test main Config class."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = Config()
        
        assert config.model.algorithm == "random_forest"
        assert config.data.target_column == "fraud"
        assert config.api.port == 8000
    
    def test_config_file_loading(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model:
  algorithm: "gradient_boosting"
  threshold: 0.7
data:
  target_column: "is_fraud"
api:
  port: 9000
""")
            config_path = f.name
        
        try:
            config = Config(config_path)
            
            assert config.model.algorithm == "gradient_boosting"
            assert config.model.threshold == 0.7
            assert config.data.target_column == "is_fraud"
            assert config.api.port == 9000
        finally:
            os.unlink(config_path)
    
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ["MODEL_ALGORITHM"] = "svm"
        os.environ["API_PORT"] = "7000"
        os.environ["LOG_LEVEL"] = "DEBUG"
        
        try:
            config = Config()
            
            assert config.model.algorithm == "svm"
            assert config.api.port == 7000
            assert config.logging.level == "DEBUG"
        finally:
            # Clean up environment variables
            os.environ.pop("MODEL_ALGORITHM", None)
            os.environ.pop("API_PORT", None)
            os.environ.pop("LOG_LEVEL", None)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Valid configuration should pass
        assert config.validate() is True
    
    def test_invalid_config_validation(self):
        """Test configuration validation with invalid values."""
        config = Config()
        
        # Set invalid values
        config.model.test_size = 1.5  # Should be between 0 and 1
        config.api.port = 100000  # Should be between 1024 and 65535
        
        # Validation should fail
        assert config.validate() is False
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert "model" in config_dict
        assert "data" in config_dict
        assert "api" in config_dict
        assert "dashboard" in config_dict
        assert "logging" in config_dict
        
        assert config_dict["model"]["algorithm"] == "random_forest"
        assert config_dict["data"]["target_column"] == "fraud"
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = Config()
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            save_path = f.name
        
        try:
            config.save(save_path)
            
            # Verify file was created
            assert Path(save_path).exists()
            
            # Load and verify content
            loaded_config = Config(save_path)
            assert loaded_config.model.algorithm == config.model.algorithm
            assert loaded_config.data.target_column == config.data.target_column
        finally:
            os.unlink(save_path)


class TestConfigEdgeCases:
    """Test configuration edge cases."""
    
    def test_nonexistent_config_file(self):
        """Test handling of nonexistent configuration file."""
        config = Config("nonexistent_file.yaml")
        
        # Should use defaults
        assert config.model.algorithm == "random_forest"
        assert config.data.target_column == "fraud"
    
    def test_empty_config_file(self):
        """Test handling of empty configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            config_path = f.name
        
        try:
            config = Config(config_path)
            
            # Should use defaults
            assert config.model.algorithm == "random_forest"
            assert config.data.target_column == "fraud"
        finally:
            os.unlink(config_path)
    
    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            config = Config(config_path)
            
            # Should use defaults despite YAML error
            assert config.model.algorithm == "random_forest"
            assert config.data.target_column == "fraud"
        finally:
            os.unlink(config_path) 