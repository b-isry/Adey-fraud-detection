"""
Data loader for the Fraud Detection System.

Handles loading data from various sources and formats with validation and error handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import logging
from abc import ABC, abstractmethod

from ..core.exceptions import DataValidationError, DataProcessingError
from ..core.logging import LoggerMixin


class BaseDataLoader(ABC, LoggerMixin):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from source."""
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate loaded data."""
        pass


class FileDataLoader(BaseDataLoader):
    """Data loader for file-based data sources."""
    
    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json',
        '.parquet': 'parquet',
        '.pickle': 'pickle',
        '.pkl': 'pickle'
    }
    
    def __init__(self, encoding: str = 'utf-8', **kwargs):
        """
        Initialize file data loader.
        
        Args:
            encoding: File encoding
            **kwargs: Additional arguments for pandas read functions
        """
        self.encoding = encoding
        self.read_kwargs = kwargs
    
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions
        
        Returns:
            Loaded DataFrame
            
        Raises:
            DataProcessingError: If file cannot be loaded
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataProcessingError(
                f"File not found: {file_path}",
                processing_step="file_loading",
                data_info={"file_path": str(file_path)}
            )
        
        # Determine file format
        file_extension = file_path.suffix.lower()
        if file_extension not in self.SUPPORTED_FORMATS:
            raise DataProcessingError(
                f"Unsupported file format: {file_extension}",
                processing_step="file_loading",
                data_info={"file_path": str(file_path), "extension": file_extension}
            )
        
        try:
            self.logger.info(f"Loading data from {file_path}")
            
            # Load data based on file format
            if file_extension in ['.csv']:
                data = pd.read_csv(
                    file_path,
                    encoding=self.encoding,
                    **{**self.read_kwargs, **kwargs}
                )
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(
                    file_path,
                    **{**self.read_kwargs, **kwargs}
                )
            elif file_extension == '.json':
                data = pd.read_json(
                    file_path,
                    encoding=self.encoding,
                    **{**self.read_kwargs, **kwargs}
                )
            elif file_extension == '.parquet':
                data = pd.read_parquet(
                    file_path,
                    **{**self.read_kwargs, **kwargs}
                )
            elif file_extension in ['.pickle', '.pkl']:
                data = pd.read_pickle(
                    file_path,
                    **{**self.read_kwargs, **kwargs}
                )
            else:
                raise DataProcessingError(
                    f"Unsupported file format: {file_extension}",
                    processing_step="file_loading"
                )
            
            self.logger.info(f"Successfully loaded data: {data.shape}")
            return data
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to load data from {file_path}: {str(e)}",
                processing_step="file_loading",
                data_info={"file_path": str(file_path), "error": str(e)}
            )
    
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate loaded data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
            
        Raises:
            DataValidationError: If data is invalid
        """
        if data is None:
            raise DataValidationError("Data is None")
        
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError("Data is not a pandas DataFrame")
        
        if data.empty:
            raise DataValidationError("Data is empty")
        
        if data.isnull().all().all():
            raise DataValidationError("All data is null")
        
        return True


class DatabaseDataLoader(BaseDataLoader):
    """Data loader for database sources."""
    
    def __init__(self, connection_string: str, **kwargs):
        """
        Initialize database data loader.
        
        Args:
            connection_string: Database connection string
            **kwargs: Additional connection parameters
        """
        self.connection_string = connection_string
        self.connection_kwargs = kwargs
    
    def load(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Load data from database using SQL query.
        
        Args:
            query: SQL query to execute
            **kwargs: Additional arguments for pandas read_sql
            
        Returns:
            Loaded DataFrame
        """
        try:
            self.logger.info("Loading data from database")
            data = pd.read_sql(
                query,
                self.connection_string,
                **{**self.connection_kwargs, **kwargs}
            )
            self.logger.info(f"Successfully loaded data: {data.shape}")
            return data
        except Exception as e:
            raise DataProcessingError(
                f"Failed to load data from database: {str(e)}",
                processing_step="database_loading",
                data_info={"query": query, "error": str(e)}
            )
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate loaded data."""
        return FileDataLoader().validate(data)


class DataLoader:
    """Main data loader class that handles multiple data sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.file_loader = FileDataLoader(**self.config.get('file_loader', {}))
        self.logger = logging.getLogger(__name__)
    
    def load_data(
        self,
        source: Union[str, Path],
        source_type: str = 'auto',
        validate: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            source: Data source (file path, database query, etc.)
            source_type: Type of source ('file', 'database', 'auto')
            validate: Whether to validate loaded data
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Loaded and validated DataFrame
        """
        try:
            # Determine source type automatically
            if source_type == 'auto':
                if isinstance(source, (str, Path)) and Path(source).exists():
                    source_type = 'file'
                elif isinstance(source, str) and source.strip().upper().startswith('SELECT'):
                    source_type = 'database'
                else:
                    source_type = 'file'  # Default to file
            
            # Load data based on source type
            if source_type == 'file':
                data = self.file_loader.load(source, **kwargs)
            elif source_type == 'database':
                # This would require a database connection
                raise NotImplementedError("Database loading not yet implemented")
            else:
                raise DataProcessingError(f"Unknown source type: {source_type}")
            
            # Validate data if requested
            if validate:
                self.file_loader.validate(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def load_multiple_files(
        self,
        file_paths: List[Union[str, Path]],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple files and return as dictionary.
        
        Args:
            file_paths: List of file paths
            **kwargs: Additional arguments for file loading
            
        Returns:
            Dictionary mapping file names to DataFrames
        """
        data_dict = {}
        
        for file_path in file_paths:
            file_name = Path(file_path).stem
            try:
                data_dict[file_name] = self.load_data(file_path, **kwargs)
                self.logger.info(f"Loaded {file_name}: {data_dict[file_name].shape}")
            except Exception as e:
                self.logger.error(f"Failed to load {file_name}: {str(e)}")
                raise
        
        return data_dict
    
    def save_data(
        self,
        data: pd.DataFrame,
        file_path: Union[str, Path],
        format: str = 'auto',
        **kwargs
    ) -> None:
        """
        Save data to file.
        
        Args:
            data: DataFrame to save
            file_path: Output file path
            format: Output format ('csv', 'excel', 'json', 'parquet', 'pickle', 'auto')
            **kwargs: Additional arguments for pandas save functions
        """
        file_path = Path(file_path)
        
        # Determine format automatically
        if format == 'auto':
            format = file_path.suffix.lower().lstrip('.')
        
        try:
            self.logger.info(f"Saving data to {file_path}")
            
            if format == 'csv':
                data.to_csv(file_path, index=False, **kwargs)
            elif format == 'excel':
                data.to_excel(file_path, index=False, **kwargs)
            elif format == 'json':
                data.to_json(file_path, orient='records', **kwargs)
            elif format == 'parquet':
                data.to_parquet(file_path, index=False, **kwargs)
            elif format == 'pickle':
                data.to_pickle(file_path, **kwargs)
            else:
                raise DataProcessingError(f"Unsupported output format: {format}")
            
            self.logger.info(f"Successfully saved data to {file_path}")
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to save data to {file_path}: {str(e)}",
                processing_step="data_saving",
                data_info={"file_path": str(file_path), "format": format, "error": str(e)}
            ) 