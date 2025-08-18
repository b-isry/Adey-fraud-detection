"""
Custom exceptions for the Fraud Detection System.

Provides specific exception types for different error scenarios.
"""


class FraudDetectionError(Exception):
    """Base exception for fraud detection system."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """
        Initialize fraud detection error.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        """String representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataValidationError(FraudDetectionError):
    """Exception raised for data validation errors."""
    
    def __init__(self, message: str, field: str = None, value: any = None):
        """
        Initialize data validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Value that failed validation
        """
        super().__init__(message, "DATA_VALIDATION_ERROR")
        self.field = field
        self.value = value
        self.details = {
            "field": field,
            "value": str(value) if value is not None else None
        }


class ModelTrainingError(FraudDetectionError):
    """Exception raised for model training errors."""
    
    def __init__(self, message: str, model_name: str = None, training_data_info: dict = None):
        """
        Initialize model training error.
        
        Args:
            message: Error message
            model_name: Name of the model that failed to train
            training_data_info: Information about the training data
        """
        super().__init__(message, "MODEL_TRAINING_ERROR")
        self.model_name = model_name
        self.training_data_info = training_data_info or {}
        self.details = {
            "model_name": model_name,
            "training_data_info": training_data_info
        }


class PredictionError(FraudDetectionError):
    """Exception raised for prediction errors."""
    
    def __init__(self, message: str, model_name: str = None, input_data_info: dict = None):
        """
        Initialize prediction error.
        
        Args:
            message: Error message
            model_name: Name of the model that failed to predict
            input_data_info: Information about the input data
        """
        super().__init__(message, "PREDICTION_ERROR")
        self.model_name = model_name
        self.input_data_info = input_data_info or {}
        self.details = {
            "model_name": model_name,
            "input_data_info": input_data_info
        }


class ConfigurationError(FraudDetectionError):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, config_key: str = None, config_value: any = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Configuration value that caused the error
        """
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key
        self.config_value = config_value
        self.details = {
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None
        }


class DataProcessingError(FraudDetectionError):
    """Exception raised for data processing errors."""
    
    def __init__(self, message: str, processing_step: str = None, data_info: dict = None):
        """
        Initialize data processing error.
        
        Args:
            message: Error message
            processing_step: Step in data processing that failed
            data_info: Information about the data being processed
        """
        super().__init__(message, "DATA_PROCESSING_ERROR")
        self.processing_step = processing_step
        self.data_info = data_info or {}
        self.details = {
            "processing_step": processing_step,
            "data_info": data_info
        }


class ModelPersistenceError(FraudDetectionError):
    """Exception raised for model saving/loading errors."""
    
    def __init__(self, message: str, operation: str = None, model_path: str = None):
        """
        Initialize model persistence error.
        
        Args:
            message: Error message
            operation: Operation that failed (save/load)
            model_path: Path to the model file
        """
        super().__init__(message, "MODEL_PERSISTENCE_ERROR")
        self.operation = operation
        self.model_path = model_path
        self.details = {
            "operation": operation,
            "model_path": model_path
        }


class APIError(FraudDetectionError):
    """Exception raised for API-related errors."""
    
    def __init__(self, message: str, status_code: int = None, endpoint: str = None):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            endpoint: API endpoint that caused the error
        """
        super().__init__(message, "API_ERROR")
        self.status_code = status_code
        self.endpoint = endpoint
        self.details = {
            "status_code": status_code,
            "endpoint": endpoint
        }


# Error code constants
ERROR_CODES = {
    "DATA_VALIDATION_ERROR": "DVE",
    "MODEL_TRAINING_ERROR": "MTE", 
    "PREDICTION_ERROR": "PE",
    "CONFIGURATION_ERROR": "CE",
    "DATA_PROCESSING_ERROR": "DPE",
    "MODEL_PERSISTENCE_ERROR": "MPE",
    "API_ERROR": "AE"
} 