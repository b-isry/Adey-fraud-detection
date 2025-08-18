#!/usr/bin/env python3
"""
Main entry point for the Fraud Detection System.

This script provides a command-line interface to run different components
of the fraud detection system.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.core.logging import setup_logging


def run_api(config: Config):
    """Run the FastAPI server."""
    import uvicorn
    from src.api.fastapi_app import create_app
    
    app = create_app(config)
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
        workers=config.api.workers
    )


def run_dashboard(config: Config):
    """Run the Streamlit dashboard."""
    import subprocess
    import os
    
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "streamlit_app.py"
    
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HOST"] = config.dashboard.host
    env["STREAMLIT_SERVER_PORT"] = str(config.dashboard.port)
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path)
    ], env=env)


def run_training(config: Config):
    """Run model training."""
    from src.models.model_factory import ModelFactory
    from src.data.data_loader import DataLoader
    
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Load data
    data_loader = DataLoader()
    # Add your data loading logic here
    
    # Train model
    model_factory = ModelFactory(config.to_dict())
    # Add your model training logic here
    
    logger.info("Model training completed.")


def run_tests():
    """Run the test suite."""
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/", "-v"
    ])
    
    return result.returncode


def validate_config(config: Config):
    """Validate configuration."""
    logger = logging.getLogger(__name__)
    
    if config.validate():
        logger.info("Configuration validation passed.")
        return True
    else:
        logger.error("Configuration validation failed.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fraud Detection System - Financial Risk Management Capstone"
    )
    
    parser.add_argument(
        "command",
        choices=["api", "dashboard", "train", "test", "validate"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Validate configuration
        if not validate_config(config):
            sys.exit(1)
        
        # Run command
        if args.command == "api":
            logger.info("Starting API server...")
            run_api(config)
        elif args.command == "dashboard":
            logger.info("Starting dashboard...")
            run_dashboard(config)
        elif args.command == "train":
            logger.info("Starting model training...")
            run_training(config)
        elif args.command == "test":
            logger.info("Running tests...")
            exit_code = run_tests()
            sys.exit(exit_code)
        elif args.command == "validate":
            logger.info("Validating configuration...")
            if validate_config(config):
                logger.info("Configuration is valid.")
                sys.exit(0)
            else:
                logger.error("Configuration is invalid.")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 