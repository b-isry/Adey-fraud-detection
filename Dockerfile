# Fraud Detection System Dockerfile
# Multi-stage build for optimized production image

# Base stage for dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed data/features

# Set permissions
RUN chmod +x main.py

# Expose ports
EXPOSE 8000 8501

# Development command
CMD ["python", "main.py", "api"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed data/features

# Set permissions
RUN chown -R appuser:appuser /app
RUN chmod +x main.py

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["python", "main.py", "api"]

# API-only stage
FROM production as api

# Only expose API port
EXPOSE 8000

# API command
CMD ["python", "main.py", "api"]

# Dashboard-only stage
FROM production as dashboard

# Only expose dashboard port
EXPOSE 8501

# Dashboard command
CMD ["python", "main.py", "dashboard"] 