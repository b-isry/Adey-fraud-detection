"""
Model factory for the Fraud Detection System.

Provides a unified interface for creating, training, and managing different ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging

from ..core.exceptions import ModelTrainingError, ConfigurationError
from ..core.logging import LoggerMixin


class BaseModel(LoggerMixin):
    """Base class for all models in the fraud detection system."""
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize base model.
        
        Args:
            name: Model name
            **kwargs: Model parameters
        """
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.training_history = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Target variable
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ModelTrainingError("Model must be trained before making predictions")
        
        if self.model is None:
            raise ModelTrainingError("Model is not initialized")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ModelTrainingError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ModelTrainingError(f"Model {self.name} does not support probability predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_names is None:
                return dict(enumerate(self.model.feature_importances_))
            return dict(zip(self.feature_names, self.model.feature_importances_))
        
        return None


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def __init__(self, **kwargs):
        """Initialize Random Forest model."""
        super().__init__("RandomForest", **kwargs)
        self.model = RandomForestClassifier(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RandomForestModel':
        """Train Random Forest model."""
        self.logger.info(f"Training {self.name} model")
        self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y, **kwargs)
        self.is_trained = True
        
        self.logger.info(f"{self.name} model training completed")
        return self


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model implementation."""
    
    def __init__(self, **kwargs):
        """Initialize Gradient Boosting model."""
        super().__init__("GradientBoosting", **kwargs)
        self.model = GradientBoostingClassifier(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'GradientBoostingModel':
        """Train Gradient Boosting model."""
        self.logger.info(f"Training {self.name} model")
        self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y, **kwargs)
        self.is_trained = True
        
        self.logger.info(f"{self.name} model training completed")
        return self


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""
    
    def __init__(self, **kwargs):
        """Initialize Logistic Regression model."""
        super().__init__("LogisticRegression", **kwargs)
        self.model = LogisticRegression(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LogisticRegressionModel':
        """Train Logistic Regression model."""
        self.logger.info(f"Training {self.name} model")
        self.feature_names = X.columns.tolist()
        
        # Scale features for logistic regression
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y, **kwargs)
        self.is_trained = True
        
        self.logger.info(f"{self.name} model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with scaled features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities with scaled features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class SVMModel(BaseModel):
    """Support Vector Machine model implementation."""
    
    def __init__(self, **kwargs):
        """Initialize SVM model."""
        super().__init__("SVM", **kwargs)
        self.model = SVC(probability=True, **kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SVMModel':
        """Train SVM model."""
        self.logger.info(f"Training {self.name} model")
        self.feature_names = X.columns.tolist()
        
        # Scale features for SVM
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y, **kwargs)
        self.is_trained = True
        
        self.logger.info(f"{self.name} model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with scaled features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities with scaled features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class NeuralNetworkModel(BaseModel):
    """Neural Network model implementation."""
    
    def __init__(self, **kwargs):
        """Initialize Neural Network model."""
        super().__init__("NeuralNetwork", **kwargs)
        self.model = MLPClassifier(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'NeuralNetworkModel':
        """Train Neural Network model."""
        self.logger.info(f"Training {self.name} model")
        self.feature_names = X.columns.tolist()
        
        # Scale features for neural network
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y, **kwargs)
        self.is_trained = True
        
        self.logger.info(f"{self.name} model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with scaled features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities with scaled features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class ModelFactory:
    """Factory class for creating and managing ML models."""
    
    AVAILABLE_MODELS = {
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel,
        'logistic_regression': LogisticRegressionModel,
        'svm': SVMModel,
        'neural_network': NeuralNetworkModel
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.models = {}
    
    def create_model(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create a new model instance.
        
        Args:
            model_type: Type of model to create
            model_name: Optional custom name for the model
            **kwargs: Model parameters
            
        Returns:
            Model instance
            
        Raises:
            ConfigurationError: If model type is not supported
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ConfigurationError(
                f"Unsupported model type: {model_type}",
                config_key="model_type",
                config_value=model_type
            )
        
        model_name = model_name or f"{model_type}_{len(self.models)}"
        model_class = self.AVAILABLE_MODELS[model_type]
        
        # Merge default config with provided kwargs
        default_params = self.config.get('default_params', {}).get(model_type, {})
        model_params = {**default_params, **kwargs}
        
        model = model_class(**model_params)
        model.name = model_name
        self.models[model_name] = model
        
        self.logger.info(f"Created {model_type} model: {model_name}")
        return model
    
    def train_model(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> BaseModel:
        """
        Train a model with optional validation.
        
        Args:
            model: Model to train
            X: Training features
            y: Training targets
            validation_data: Optional validation data
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        try:
            self.logger.info(f"Training model: {model.name}")
            
            # Train the model
            model.fit(X, y, **kwargs)
            
            # Validate if validation data provided
            if validation_data:
                X_val, y_val = validation_data
                val_predictions = model.predict(X_val)
                val_accuracy = (val_predictions == y_val).mean()
                self.logger.info(f"Validation accuracy: {val_accuracy:.4f}")
            
            return model
            
        except Exception as e:
            raise ModelTrainingError(
                f"Failed to train model {model.name}: {str(e)}",
                model_name=model.name,
                training_data_info={"X_shape": X.shape, "y_shape": y.shape}
            )
    
    def hyperparameter_tuning(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = 'accuracy',
        search_type: str = 'grid',
        n_iter: int = 100,
        **kwargs
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Perform hyperparameter tuning.
        
        Args:
            model_type: Type of model to tune
            X: Training features
            y: Training targets
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            scoring: Scoring metric
            search_type: Type of search ('grid' or 'random')
            n_iter: Number of iterations for random search
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (best_model, best_params)
        """
        self.logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        # Create base model
        base_model = self.create_model(model_type, **kwargs)
        
        # Create search object
        if search_type == 'grid':
            search = GridSearchCV(
                base_model.model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                base_model.model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ConfigurationError(f"Unsupported search type: {search_type}")
        
        # Perform search
        search.fit(X, y)
        
        # Create best model
        best_model = self.create_model(
            model_type,
            model_name=f"{model_type}_best",
            **search.best_params_
        )
        best_model.fit(X, y)
        
        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best score: {search.best_score_:.4f}")
        
        return best_model, search.best_params_
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """
        Get a model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance or None if not found
        """
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from the factory.
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            True if model was removed, False if not found
        """
        if model_name in self.models:
            del self.models[model_name]
            self.logger.info(f"Removed model: {model_name}")
            return True
        return False 